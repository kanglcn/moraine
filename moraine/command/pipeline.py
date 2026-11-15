"""Run a processing pipeline described in a TOML file, resuming from where it stopped.

Example ``pipeline.toml``::

    [pipeline]
    version = 1            # pipeline file format version (optional, default 1)
    workdir = "."          # relative paths are relative to this directory (default: the TOML directory);
                           # `moraine run --workdir DIR` overrides it
    quicklook = true       # save a PNG of every pyramid made by a step (default: true)

    [vars]                 # ${name} in any value is replaced; `moraine run --var gamma=/data/gamma` overrides
    gamma = "/path/to/gamma"
    dates = { file = "raw/meta.toml", key = "dates" }   # a top-level key of a TOML file in the working directory;
                           # a value that is exactly "${dates}" keeps the type of the variable (here a list)

    [defaults]             # applied to every step whose command has these arguments
    cuda = true

    [[step]]
    name = "adi"           # unique step name
    run = "amp-disp"       # a command from `moraine list`
    rslc = "raw/rslc.zarr" # the other keys are the command arguments
    adi = "ps/adi.zarr"

    [[step]]
    name = "low_adi"
    run = "math"
    output = "ps/low_adi.zarr"
    operation = "adi < 0.4"
    [step.kw]              # extra keyword arguments: the input arrays of `math`
    adi = "ps/adi.zarr"

State, logs, output metadata and quicklooks of pyramids are written to ``<workdir>/.moraine/<file name>/``,
so several pipeline files can share a working directory. A step is skipped when it
finished before with the same arguments and all its outputs still exist.
"""

__all__ = ['load_pipeline', 'run_pipeline', 'pipeline_status']

import contextlib
import difflib
import hashlib
import json
import logging
import os
import re
import sys
import time
import tomllib
import traceback
from pathlib import Path

from . import UsageError, get_command, bind_args, execute, _print_summary, _strings

_RESERVED = {'name', 'run', 'quicklook', 'kw'}
# version of the pipeline file format, see docs/contracts/pipeline-file.md
PIPELINE_VERSION = 1
_VAR = re.compile(r'\$\{(\w+)\}')
_FILE_VAR = {'file', 'key'}       # [vars] NAME = { file = "PATH", key = "KEY" }: a top-level key of a TOML file


class _Pending(Exception):
    """A variable read from a file that does not exist yet: the steps using it wait for the file."""


def _substitute(value, variables, where, used):
    """Replace ${name} in strings (also inside lists and tables) and add the names to `used`.

    A string that is exactly ${name} becomes the value of the variable with its type (a list stays a list); inside a
    longer string ${name} is replaced by the text of the value. Raises _Pending for a variable whose file does not
    exist yet.
    """
    if isinstance(value, str):
        names = _VAR.findall(value)
        for n in names:
            if n not in variables:
                known = ', '.join(sorted(variables)) or 'none'
                raise UsageError(f'{where}: unknown variable ${{{n}}}; defined: {known} '
                                 '(add it to [vars] or pass --var NAME=VALUE)')
        used.update(names)
        for n in names:
            if isinstance(variables[n], _Pending):
                raise _Pending(str(variables[n]))
        whole = _VAR.fullmatch(value)
        if whole:
            return variables[whole.group(1)]

        def text(m):
            v = variables[m.group(1)]
            if isinstance(v, (list, dict)):
                kind = 'list' if isinstance(v, list) else 'table'
                raise UsageError(f'{where}: variable ${{{m.group(1)}}} is a {kind} and can only be a whole value, '
                                 f'e.g. "${{{m.group(1)}}}"')
            return str(v)
        return _VAR.sub(text, value)
    if isinstance(value, list):
        return [_substitute(v, variables, where, used) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, variables, where, used) for k, v in value.items()}
    return value


def _var_specs(cfg_vars:dict, overrides:dict, path)->dict:
    """The [vars] of a pipeline file with the --var overrides; a table is a variable read from a file."""
    for name, v in cfg_vars.items():
        if isinstance(v, dict) and (set(v) != _FILE_VAR or not all(isinstance(x, str) for x in v.values())):
            raise UsageError(f'{path}: [vars] {name} = {v}: a variable read from a file is written '
                             '{ file = "PATH", key = "KEY" }, the top-level key KEY of the TOML file PATH')
    return {**cfg_vars, **(overrides or {})}


def _read_vars(specs:dict, workdir, path)->dict:
    """Values of the variables; one whose file does not exist yet is a _Pending."""
    values, files = {}, {}
    for name, v in specs.items():
        if not isinstance(v, dict):
            values[name] = v
            continue
        f = Path(workdir) / v['file']
        if not f.exists():
            values[name] = _Pending(f'variable {name}: {v["file"]} does not exist yet')
            continue
        if f not in files:
            try:
                files[f] = tomllib.loads(f.read_text())
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
                raise UsageError(f'{path}: [vars] {name}: {v["file"]} is not a readable TOML file: {e}')
        if v['key'] not in files[f]:
            keys = list(files[f])
            close = difflib.get_close_matches(v['key'], keys, n=1)
            hint = f' Did you mean {close[0]!r}?' if close else ''
            raise UsageError(f'{path}: [vars] {name}: no key {v["key"]!r} in {v["file"]}.{hint} '
                             f'Keys: {", ".join(keys) or "none"}')
        values[name] = files[f][v['key']]
    return values


def _step(raw:dict, pipe:dict, variables:dict)->dict:
    """A step from its table in the pipeline file: the variables substituted, the arguments checked, its hash.

    A step using a variable whose file does not exist yet gets None for the values with it (the names of the
    arguments are checked all the same) and the reason in 'pending'; it has to run.
    """
    path, name = pipe['file'], raw['name']
    cmd = get_command(raw['run'])
    used, pending = set(), []

    def sub(value, where):
        try:
            return _substitute(value, variables, where, used)
        except _Pending as e:
            pending.append(str(e))
            return None

    where = f'{path}: step {name!r}'
    values = {k: sub(v, f'{path}: [defaults]') for k, v in pipe['defaults'].items()
              if any(p.name == k for p in cmd.params)}
    values.update({k: sub(v, where) for k, v in raw.items() if k not in _RESERVED})
    kw = raw.get('kw') or {}
    if not isinstance(kw, dict):
        raise UsageError(f'{where}: kw must be a table ([step.kw]), got {kw!r}')
    kw = {k: sub(v, where) for k, v in kw.items()}
    quicklook = sub(raw['quicklook'], where) if 'quicklook' in raw else pipe['meta'].get('quicklook', True)
    try:
        kwargs = bind_args(cmd, values, kw)
    except UsageError as e:
        raise UsageError(f'{where}: {e}')
    if kw:
        values = {**values, 'kw': kw}   # part of the step hash
    return {'name': name, 'command': cmd, 'values': values, 'kwargs': kwargs, 'quicklook': quicklook,
            'hash': hashlib.sha1(json.dumps([cmd.name, values], sort_keys=True, default=str).encode()).hexdigest()[:12],
            'pending': pending[0] if pending else None,
            # files of the variables used: read again before the step runs, an earlier step may make or change them
            'var_files': sorted({pipe['vars'][n]['file'] for n in used if isinstance(pipe['vars'][n], dict)}),
            'raw': raw}


def load_pipeline(path:str, workdir:str=None, variables:dict=None)->dict:
    """Read and check a pipeline file. Returns {'file', 'name', 'workdir', 'steps': [...], ...}.

    `workdir` overrides [pipeline] workdir and `variables` override [vars].
    """
    path = Path(path)
    if not path.is_file():
        raise UsageError(f'pipeline file {path} not found')
    try:
        cfg = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise UsageError(f'{path}: invalid TOML: {e}')
    unknown = set(cfg) - {'pipeline', 'vars', 'defaults', 'step'}
    if unknown:
        raise UsageError(f'{path}: unknown table(s) {sorted(unknown)}; expected [pipeline], [vars], [defaults], [[step]]')
    meta = cfg.get('pipeline', {})
    unknown = set(meta) - {'version', 'workdir', 'quicklook'}
    if unknown:
        raise UsageError(f'{path}: unknown key(s) {sorted(unknown)} in [pipeline]; expected version, workdir, quicklook')
    version = meta.get('version', 1)
    if not isinstance(version, int) or version < 1 or version > PIPELINE_VERSION:
        raise UsageError(f'{path}: pipeline format version {version!r} is not supported by this moraine '
                         f'(supported: 1..{PIPELINE_VERSION}); update moraine')
    workdir = Path(workdir).resolve() if workdir else (path.parent / meta.get('workdir', '.')).resolve()
    pipe = {'file': str(path), 'name': path.stem, 'workdir': workdir, 'meta': meta,
            'vars': _var_specs(cfg.get('vars', {}), variables, path), 'defaults': cfg.get('defaults', {})}
    values = _read_vars(pipe['vars'], workdir, path)
    try:
        _substitute(pipe['defaults'], values, f'{path}: [defaults]', set())   # unknown variables, also if unused
    except _Pending:
        pass
    steps, names = [], set()
    for i, raw in enumerate(cfg.get('step', []), 1):
        name = raw.get('name')
        if not name:
            raise UsageError(f'{path}: step {i} has no name')
        if name in names:
            raise UsageError(f'{path}: duplicated step name {name!r}')
        names.add(name)
        if 'run' not in raw:
            raise UsageError(f'{path}: step {name!r} has no `run = "<command>"`')
        steps.append(_step(raw, pipe, values))
    if not steps:
        raise UsageError(f'{path}: no [[step]] defined')
    pipe['steps'] = steps
    return pipe


def _home(pipe):
    """Directory of the state, logs and quicklooks of a pipeline: <workdir>/.moraine/<file name>/."""
    return Path(pipe['workdir']) / '.moraine' / pipe['name']


def _load_state(pipe):
    f = _home(pipe) / 'state.json'
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(pipe, state):
    f = _home(pipe) / 'state.json'
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(f)


def _why_rerun(step, rec, workdir):
    """Reason why `step` has to run, or None when its previous run is still valid."""
    from . import _mtime
    if step['pending']:
        return step['pending']
    if rec is None:
        return 'not run yet'
    if rec.get('status') != 'done':
        return f'previous run {rec.get("status")}'
    if rec.get('hash') != step['hash']:
        return 'arguments changed'
    missing = [o for o in rec.get('outputs', []) if not (Path(workdir) / o).exists()]
    if missing:
        return f'output missing: {missing[0]}'
    changed = [i for i, t in rec.get('inputs', {}).items() if _mtime(Path(workdir) / i) != t]
    if changed:
        return f'input changed: {changed[0]}'
    return None


@contextlib.contextmanager
def _in_dir(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _plan(pipe, state, only=None, from_step=None, force=False):
    names = [s['name'] for s in pipe['steps']]
    for n in (only or []) + ([from_step] if from_step else []):
        if n not in names:
            raise UsageError(f'unknown step {n!r}; steps: {", ".join(names)}')
    plan, rerun, will_change = [], False, set()
    for s in pipe['steps']:
        rec = state.get(s['name'])
        if from_step and s['name'] == from_step:
            rerun = True
        why = _why_rerun(s, rec, pipe['workdir'])
        # the file of a variable is read by the steps using it
        upstream = [i for i in [*(rec or {}).get('inputs', {}), *s['var_files']] if i in will_change]
        if only and s['name'] not in only:
            action = 'skip (not selected)'
        elif force or rerun:
            action = 'run (forced)'
        elif why:
            action = f'run ({why})'
        elif upstream:
            action = f'run (input will change: {upstream[0]})'
        else:
            action = 'skip (done)'
        if action.startswith('run'):
            # outputs of a step that runs change the steps reading them
            will_change.update((rec or {}).get('outputs') or [v for v in _strings(s['kwargs'].values())])
        plan.append((s, action))
    return plan


def pipeline_status(path:str, workdir:str=None, variables:dict=None)->dict:
    """State of every step of a pipeline: done / failed / changed / pending."""
    pipe = load_pipeline(path, workdir, variables)
    state = _load_state(pipe)
    steps = []
    for s, action in _plan(pipe, state):
        rec = state.get(s['name'])
        if rec is None:
            status = 'pending'
        elif action == 'skip (done)':
            status = 'done'
        elif rec.get('status') == 'done':
            status = 'outdated, will ' + action
        else:
            status = rec.get('status', 'pending')
        item = {'name': s['name'], 'command': s['command'].name, 'status': status}
        for k in ('seconds', 'finished', 'error', 'log', 'outputs', 'quicklooks'):
            if rec and k in rec:
                item[k] = rec[k]
        steps.append(item)
    return {'pipeline': pipe['file'], 'workdir': str(pipe['workdir']), 'steps': steps}


def run_pipeline(path:str, only:list=None, from_step:str=None, force:bool=False,
                 dry_run:bool=False, quicklook:bool=True, echo=print, workdir:str=None, variables:dict=None)->dict:
    """Run the steps of a pipeline file that are not done yet. Stops at the first failing step."""
    from .summary import quicklook as _quicklook
    pipe = load_pipeline(path, workdir, variables)
    workdir = pipe['workdir']
    state = _load_state(pipe)
    plan = _plan(pipe, state, only, from_step, force)
    result = {'pipeline': pipe['file'], 'workdir': str(workdir), 'ok': True,
              'plan': [{'name': s['name'], 'command': s['command'].name, 'action': a} for s, a in plan], 'steps': []}
    if dry_run:
        for s, a in plan:
            echo(f'{s["name"]:24s} {s["command"].name:28s} {a}')
        return result
    workdir.mkdir(parents=True, exist_ok=True)
    logdir = _home(pipe) / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for s, action in plan:
        problem = None
        if s['var_files'] and action != 'skip (not selected)':
            try:      # the files of its variables are read again: an earlier step may have made or changed them
                s = _step(s['raw'], pipe, _read_vars(pipe['vars'], workdir, pipe['file']))
            except UsageError as e:
                problem = str(e)
        if action == 'skip (done)':
            why = problem or _why_rerun(s, state.get(s['name']), workdir)   # an upstream step may have rewritten an input
            if why:
                action = f'run ({why})'
        if not action.startswith('run'):
            echo(f'[{s["name"]}] {action}')
            continue
        echo(f'[{s["name"]}] running {s["command"].name} ...')
        log = logdir / f'{s["name"]}.log'
        handler = logging.FileHandler(log, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
        handler.setLevel(logging.INFO)
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(min(old_level, logging.INFO))
        rec = {'command': s['command'].name, 'hash': s['hash'], 'args': s['values'],
               'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'log': str(log.relative_to(workdir))}
        try:
            if problem or s['pending']:
                raise UsageError(problem or s['pending'])
            with _in_dir(workdir):
                out = execute(s['command'], s['kwargs'])
                rec.update(status='done', seconds=out['seconds'], outputs=out['outputs'], inputs=out['inputs'],
                           summaries=out['summaries'])
                if s['quicklook'] and quicklook:
                    rec['quicklooks'] = []
                    for summ in out['summaries']:
                        if not summ.get('kind', '').endswith('pyramid'):   # only pyramids are drawn
                            continue
                        png = _home(pipe).relative_to(workdir) / 'quicklook' / f'{s["name"]}__{Path(summ["path"].rstrip("/")).name}.png'
                        try:
                            rec['quicklooks'].append(_quicklook(summ['path'], str(png)))
                        except Exception as e:
                            logging.getLogger(__name__).warning(f'quicklook of {summ["path"]} failed: {e}')
        except Exception as e:
            rec.update(status='failed', error=f'{type(e).__name__}: {e}',
                       traceback=traceback.format_exc(limit=-8))
            logging.getLogger(__name__).error(traceback.format_exc())
        finally:
            rec['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            root.removeHandler(handler); handler.close(); root.setLevel(old_level)
            state[s['name']] = rec
            _save_state(pipe, state)
        result['steps'].append({'name': s['name'], **{k: v for k, v in rec.items() if k != 'traceback'}})
        if rec['status'] == 'failed':
            result['ok'] = False
            echo(f'[{s["name"]}] FAILED: {rec["error"]}')
            echo(f'  log: {workdir / rec["log"]}')
            echo(f'  fix the problem, then rerun `moraine run {pipe["file"]}` to resume from this step')
            break
        echo(f'[{s["name"]}] done in {rec["seconds"]} s')
        for summ in rec['summaries']:
            _print_summary(summ, echo)
    return result


def _vars(args):
    out = {}
    for item in args.var or []:
        name, sep, value = item.partition('=')
        if not sep or not name:
            raise UsageError(f'--var expects NAME=VALUE, got {item!r}')
        out[name] = value
    return out


def cli(args, emit):
    """`moraine run` and `moraine status`."""
    if args._sub == 'status':
        st = pipeline_status(args.pipeline, args.workdir, _vars(args))
        def text():
            print(f'workdir: {st["workdir"]}')
            for s in st['steps']:
                extra = f'  {s["seconds"]} s' if 'seconds' in s else ''
                print(f'  {s["name"]:24s} {s["command"]:28s} {s["status"]}{extra}')
                if 'error' in s and s['status'] == 'failed':
                    print(f'      error: {s["error"]}\n      log: {s.get("log")}')
        return emit(args, st, text)
    echo = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print
    result = run_pipeline(args.pipeline, only=args.only, from_step=args.from_step, force=args.force,
                          dry_run=args.dry_run, quicklook=not args.no_quicklook, echo=echo,
                          workdir=args.workdir, variables=_vars(args))
    emit(args, result)
    return 0 if result['ok'] else 1
