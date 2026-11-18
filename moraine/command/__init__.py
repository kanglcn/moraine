"""The ``moraine`` command line.

Every logged function of ``moraine.cli`` is a sub command, with its options and help generated from the
function signature and numpy docstring::

    moraine list                                  # all commands
    moraine amp-disp --help                       # options of one command
    moraine amp-disp --rslc raw/rslc.zarr --adi ps/adi.zarr --cuda
    moraine info ps/adi.zarr                      # shape, dtype and chunks of a result
    moraine ras-pyramid --ras ps/adi.zarr --out_dir ps/adi_pyramid
    moraine quicklook ps/adi_pyramid -o adi.png   # PNG of a pyramid, drawn like moraine.cli.view
    moraine view ps/adi_pyramid -o view.ipynb     # notebook with interactive maps of pyramids (moraine.cli.view)
    moraine image-pairs --rslc raw/rslc.zarr --bandwidth 1 --out pairs.txt
    moraine run pipeline.toml                     # run (or resume) a processing pipeline
    moraine status pipeline.toml

Add ``--json`` to any command for machine readable output on stdout; logs always go to stderr.
"""

__all__ = ['main', 'commands', 'JSON_VERSION']

import argparse
import ast
import contextlib
import difflib
import functools
import importlib
import inspect
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..api.tnet import check_image_pairs, _integer_image_pairs
from ..api.utils_.errors import ArgumentError

# modules of moraine.cli whose logged functions become commands
_MODULES = ['load', 'transform', 'tnet', 'math', 'polygon', 'pc', 'ps', 'shp', 'co', 'pl', 'dl', 'pqm', 'pu', 'tropo', 'tide', 'iono', 'ts', 'export', 'plot']
# version of the --json output format, see docs/contracts/json-output.md
JSON_VERSION = 1


class UsageError(Exception):
    """Wrong command line or pipeline input."""


# ---------------------------------------------------------------- docstrings

def parse_docstring(doc:str)->tuple:
    """Split a numpy docstring into (summary, {parameter: (type, description)})."""
    lines = inspect.cleandoc(doc or '').splitlines()
    summary, params, i = [], {}, 0
    while i < len(lines) and not (i + 1 < len(lines) and set(lines[i + 1].strip()) == {'-'}):
        summary.append(lines[i]); i += 1
    while i < len(lines):
        header = lines[i].strip(); i += 2
        body = []
        while i < len(lines) and not (i + 1 < len(lines) and set(lines[i + 1].strip()) == {'-'} and lines[i + 1].strip()):
            body.append(lines[i]); i += 1
        if header == 'Parameters':
            # parameter names are indented like the section header, descriptions deeper
            base = len(lines[i - len(body) - 2]) - len(lines[i - len(body) - 2].lstrip())
            name = None
            for line in body:
                indent = len(line) - len(line.lstrip())
                if line.strip() and indent <= base:
                    name, _, typ = line.partition(' : ')
                    name = name.lstrip('*').strip()
                    params[name] = [typ.strip(), '']
                elif name and line.strip():
                    params[name][1] = (params[name][1] + ' ' + line.strip()).strip()
    return ' '.join(l.strip() for l in summary if l.strip()), {k: tuple(v) for k, v in params.items()}


# ---------------------------------------------------------------- value parsing

def literal(text):
    """Python literal if possible ('1000,1000' -> (1000, 1000), 'None' -> None), the text itself otherwise."""
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def image_pairs(value):
    """Image pairs from a list, a python literal, or a text/.npy file with two columns (ref, sec) of integers, checked
    against the data conventions (sorted without duplicates, ref < sec); int64 (n_pairs, 2). UsageError otherwise,
    also for a file that numpy cannot read."""
    where = '' if isinstance(value, (np.ndarray, list, tuple)) else f' {str(value)!r}'
    try:
        if isinstance(value, (np.ndarray, list, tuple)):
            pairs = np.asarray(value)
        else:
            text = str(value)
            if Path(text).is_file():
                # as floats: values that are not integers are errors below, which reading as int64 truncated (0.5 to 0)
                pairs = np.load(text) if text.endswith('.npy') else np.loadtxt(text, ndmin=2)
            else:
                parsed = literal(text)
                if isinstance(parsed, str):
                    raise UsageError(f'image pairs {text!r}: not a file and not a list like [[0,1],[1,2]]; '
                                     'make a file with `moraine image-pairs`')
                pairs = np.asarray(parsed)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise UsageError(f'image pairs must have shape (n, 2), got {pairs.shape}')
        pairs = _integer_image_pairs(pairs)
        check_image_pairs(pairs)
    except (ValueError, OSError, EOFError) as e:    # also a file numpy cannot read, lists of different lengths
        raise UsageError(f'image pairs{where}: {e}') from None
    return pairs


def _tuple_spec(annotation:str)->tuple:
    """(number of integers or None, single int accepted) from e.g. 'tuple[int, int]' or 'int | tuple[int, int]'."""
    m = re.search(r'tuple\[([^\]]*)\]', annotation)
    n = None if not m or '...' in m.group(1) else len([t for t in m.group(1).split(',') if t.strip()])
    scalar = bool(re.search(r'(^|\|)\s*int\s*(\||$)', annotation))
    return n, scalar


def _int_tuple(p, value):
    """Parse '2500 1834' (two tokens), '2500,1834', '(2500,1834)', [2500, 1834] or 2500 and check it against `p`;
    None (also the text None) is the default of an optional tuple and a UsageError for a required one."""
    raw = value
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        tokens = [t for v in value for t in v.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
                  .replace(',', ' ').split()]                    # command line tokens
        value = None if tokens in (['None'], []) else tuple(literal(t) for t in tokens)
        if value is not None and len(value) == 1 and p.scalar:
            value = value[0]
    elif isinstance(value, str):
        value = literal(value)
    what = f'{p.n} integers' if p.n else 'integers'
    if p.scalar:
        what = f'an integer or {what}'
    hint = f' ({p.help})' if p.help else ''
    shown = ' '.join(raw) if isinstance(raw, (list, tuple)) and all(isinstance(v, str) for v in raw) else raw
    example = f'write e.g. --{p.name} ' + ' '.join(['1000'] * (p.n or 2))
    if value is None:
        if p.required:
            raise UsageError(f'--{p.name} is required: it needs {what}{hint}, got {shown!r}; {example}')
        return None
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if p.scalar:
            return value
        value = (value,)
    if not isinstance(value, tuple) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value) \
       or (p.n and len(value) != p.n):
        raise UsageError(f'--{p.name} needs {what}{hint}, got {shown!r}; {example}')
    return value


# ---------------------------------------------------------------- commands

@dataclass
class Param:
    name: str
    kind: str          # 'str', 'bool', 'list', 'pairs', 'value'
    required: bool
    default: object
    type_doc: str
    help: str
    n: int = None          # 'tuple': required number of integers (None: any)
    scalar: bool = False   # 'tuple': a single integer is accepted too (int | tuple[...])


@dataclass
class Command:
    name: str          # command name, e.g. 'amp-disp'
    func: object       # the logged moraine.cli function
    module: str
    summary: str
    params: list = field(default_factory=list)
    has_kwargs: bool = False
    kwargs_help: str = ''

    def convert(self, name, value):
        """Convert a value from the command line or a pipeline file to what the function expects; the text None is
        None, the default of an optional argument, and a UsageError for a required one, as is an empty list for a
        required list argument."""
        p = next(q for q in self.params if q.name == name)
        if value is None:
            return None             # from Python, or a pipeline variable of a file not made yet
        if p.kind == 'pairs':
            if isinstance(value, str) and literal(value) is None:
                return self._none(p)
            if isinstance(value, str) and isinstance(literal(value), str):
                return value            # a file, read in `execute`: it may be made by an earlier pipeline step
            return image_pairs(value)
        if p.kind == 'list':
            if isinstance(value, (list, tuple)):
                if not value:           # e.g. times = [] in a pipeline file
                    if p.required:
                        raise UsageError(f'{self.name}: argument {name} needs at least one value, got an empty '
                                         f'list: {p.help}')
                    return []           # the command decides, e.g. pc-sort with pc_in = [] and pc = []: no data
                return list(value) if len(value) > 1 else value[0]
            return value
        if p.kind == 'str':
            return str(value)
        if p.kind == 'tuple':
            return _int_tuple(p, value)
        if p.kind == 'bool':
            if isinstance(value, str):      # text from `--var` or a pipeline variable: bool('false') would be True
                if value.strip().lower() not in ('true', 'false'):
                    raise UsageError(f'{self.name}: argument {name} = {value!r} is not a boolean (true or false)')
                return value.strip().lower() == 'true'
            return bool(value)
        value = literal(value)
        if value is None:
            return self._none(p)
        if isinstance(value, str) and (p.type_doc or '').split(',')[0].strip() in ('float', 'int'):
            hint = ' (an empty value: a --var NAME= without a value?)' if value == '' else ''
            raise UsageError(f'{self.name}: argument {name} = {value!r} is not a number{hint}')
        if isinstance(value, list) and 'tuple' in p.type_doc:
            value = tuple(value)
        return value

    def _none(self, p):
        """None for an optional argument; a required argument has no default and None is not one of its values"""
        if p.required:
            raise UsageError(f'{self.name}: argument {p.name} is required and cannot be None: {p.help}')
        return None


def _kind(p):
    a = p.annotation
    if isinstance(a, str):
        text = a
    elif hasattr(a, '__args__'):            # tuple[int, int], int | tuple[int, int], str | list
        text = str(a).replace('typing.', '')
    else:
        text = getattr(a, '__name__', str(a))
    if 'ndarray' in text:
        return 'pairs', text
    if text == 'bool' or isinstance(p.default, bool):
        return 'bool', text
    if 'list' in text and 'str' in text:
        return 'list', text
    if 'tuple' in text:
        return 'tuple', text
    if text == 'str':
        return 'str', text
    return 'value', text if a is not inspect.Parameter.empty else ''


@functools.lru_cache(maxsize=None)
def commands()->dict:
    """All commands, {name: Command}."""
    out = {}
    for mod in _MODULES:
        m = importlib.import_module(f'moraine.cli.{mod}')
        for fname in getattr(m, '__all__', []):
            f = getattr(m, fname)
            raw = getattr(f, '__wrapped__', None)
            if raw is None:
                continue
            summary, docs = parse_docstring(raw.__doc__)
            cmd = Command(fname.replace('_', '-'), f, mod, summary)
            for p in inspect.signature(raw).parameters.values():
                if p.kind is inspect.Parameter.VAR_KEYWORD:
                    cmd.has_kwargs = True
                    cmd.kwargs_help = docs.get(p.name, ('', ''))[1] or 'passed to the function as keyword arguments'
                    continue
                kind, text = _kind(p)
                required = p.default is inspect.Parameter.empty
                typ, desc = docs.get(p.name, (text, ''))
                n, scalar = _tuple_spec(text) if kind == 'tuple' else (None, False)
                cmd.params.append(Param(p.name, kind, required, None if required else p.default, typ or text, desc,
                                        n, scalar))
            out[cmd.name] = cmd
    return out


def get_command(name:str)->Command:
    cmds = commands()
    key = name.replace('_', '-')
    if key not in cmds:
        close = difflib.get_close_matches(key, cmds, n=3)
        hint = f' Did you mean: {", ".join(close)}?' if close else ' Run `moraine list` for all commands.'
        raise UsageError(f'unknown command {name!r}.{hint}')
    return cmds[key]


def bind_args(cmd:Command, values:dict, kw:dict=None)->dict:
    """Check and convert arguments (from a pipeline file or the command line) for `cmd`.

    Unknown names are errors, so a typo is never passed silently to ``**kwargs``; extra keyword
    arguments (e.g. the input arrays of `math`) have to be given explicitly in `kw`.
    """
    names = [p.name for p in cmd.params]
    unknown = [k for k in values if k not in names]
    if unknown:
        hints = [f'{k} -> {c[0]}' for k in unknown if (c := difflib.get_close_matches(k, names, n=1))]
        hint = f' Did you mean: {", ".join(hints)}?' if hints else ''
        raise UsageError(f'{cmd.name}: unknown argument(s): {", ".join(unknown)}.{hint} '
                         f'Valid arguments: {", ".join(names)}')
    missing = [p.name for p in cmd.params if p.required and p.name not in values]
    if missing:
        raise UsageError(f'{cmd.name}: missing required argument(s): {", ".join(missing)}')
    kwargs = {k: cmd.convert(k, v) for k, v in values.items()}
    if kw:
        if not cmd.has_kwargs:
            raise UsageError(f'{cmd.name} takes no extra keyword arguments')
        clash = [k for k in kw if k in names]
        if clash:
            raise UsageError(f'{cmd.name}: {", ".join(clash)} are normal arguments, not extra keyword arguments')
        kwargs.update({k: literal(v) for k, v in kw.items()})
    return kwargs


# ---------------------------------------------------------------- running

def _mtime(path):
    p = Path(path)
    if not p.exists():
        return None
    times = [p.stat().st_mtime_ns]
    for meta in ('zarr.json', '.zarray', '.zgroup', '.zattrs'):
        if (p / meta).exists():
            times.append((p / meta).stat().st_mtime_ns)
    return max(times)


def _strings(values):
    for v in values:
        if isinstance(v, str):
            yield v
        elif isinstance(v, (list, tuple)):
            yield from _strings(v)


def execute(cmd:Command, kwargs:dict, summarize_outputs:bool=True)->dict:
    """Run `cmd` and return a record with the outputs it created or modified and their summaries."""
    from .summary import summarize
    # any string may be a path, e.g. out_dir='pyr'; but not '', which pathlib takes for the working directory
    candidates = list(dict.fromkeys(s for s in _strings(kwargs.values()) if s))
    before = {s: _mtime(s) for s in candidates}
    call = dict(kwargs)
    for p in cmd.params:              # image pair files are read now, relative to the working directory
        if p.kind == 'pairs' and isinstance(call.get(p.name), str):
            call[p.name] = image_pairs(call[p.name])
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):  # nothing a command prints may break the --json output
        cmd.func(**call)
    record = {'command': cmd.name, 'seconds': round(time.time() - t0, 1), 'outputs': [], 'inputs': {}}
    for s in candidates:
        after = _mtime(s)
        if after is not None and after != before[s]:
            record['outputs'].append(s)
        elif after is not None:
            record['inputs'][s] = after  # read, not written: remember its modification time
    if summarize_outputs:
        record['summaries'] = []
        for s in record['outputs']:
            try:
                record['summaries'].append(summarize(s))
            except Exception as e:  # a summary must never hide a successful run
                record['summaries'].append({'path': s, 'error': f'{type(e).__name__}: {e}'})
    return record


# ---------------------------------------------------------------- output

def _setup_logging(args):
    root = logging.getLogger()
    root.handlers.clear()
    level = logging.WARNING if getattr(args, 'quiet', False) else logging.INFO
    root.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    h = logging.StreamHandler(sys.stderr); h.setFormatter(fmt); h.setLevel(level); root.addHandler(h)
    if getattr(args, 'log', None):
        fh = logging.FileHandler(args.log); fh.setFormatter(fmt); fh.setLevel(logging.INFO); root.addHandler(fh)
        root.setLevel(logging.INFO)
    for noisy in ('numba', 'matplotlib', 'fsspec', 'asyncio'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_summary(s, echo=print):
    if 'error' in s:
        echo(f'  {s["path"]}: summary failed ({s["error"]})'); return
    if 'shape' not in s:
        rest = {k: v for k, v in s.items() if k not in ('path', 'kind')}
        echo(f'  {s["path"]}: {s.get("kind")} {rest}'); return
    head = f'  {s["path"]}: {s["dtype"]} {tuple(s["shape"])}'
    head += f' chunks {tuple(s["chunks"])}' if 'chunks' in s else f' {s["kind"]}, {s["levels"]} levels'
    stats = {k: v for k, v in s.items() if k not in ('path', 'kind', 'shape', 'dtype', 'chunks', 'levels', 'warnings')}
    echo(head)
    if stats:
        echo('    ' + ', '.join(f'{k}={v}' for k, v in stats.items()))
    for w in s.get('warnings', []):
        echo(f'    WARNING: {w}')


def _emit(args, result, text=None):
    if args.json:
        result = {'version': JSON_VERSION, 'ok': True, **result}
        print(json.dumps(result, default=str))
    elif text:
        text()


# ---------------------------------------------------------------- argument parser

def _add_global(p):
    g = p.add_argument_group('output')
    g.add_argument('--json', action='store_true', help='print a machine readable JSON result on stdout')
    g.add_argument('--traceback', action='store_true', help='show the full traceback on errors')
    g.add_argument('--log', metavar='FILE', help='also write the log to FILE')
    g.add_argument('-q', '--quiet', action='store_true', help='only log warnings and errors')


def _add_command_parser(sub, cmd:Command):
    p = sub.add_parser(cmd.name, help=cmd.summary.split('. ')[0][:80], description=cmd.summary,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    req = p.add_argument_group('required arguments')
    opt = p.add_argument_group('optional arguments')
    for q in cmd.params:
        flags = [f'--{q.name}']   # --a-b is accepted too, see _normalize_options
        help_ = (q.help or '') + (f' [{q.type_doc}]' if q.type_doc else '')
        if not q.required and q.kind != 'bool' and not any(w in q.type_doc for w in ('default', 'optional')):
            help_ += f' (default: {q.default!r})'
        group = req if q.required else opt
        if q.kind == 'bool':
            group.add_argument(*flags, dest=q.name, action=argparse.BooleanOptionalAction,
                               default=argparse.SUPPRESS, help=help_)
        elif q.kind == 'list':
            # paths of arrays (the help names them or says input / output) or other strings, e.g. dates
            paths = re.search(r'\bpaths?\b|^(input|output)\b', q.help or '', re.I)
            group.add_argument(*flags, dest=q.name, nargs='+', required=q.required, default=argparse.SUPPRESS,
                               metavar='PATH' if paths else 'STR', help=help_ + ' (one or more)')
        elif q.kind == 'tuple':
            group.add_argument(*flags, dest=q.name, nargs='+', required=q.required, default=argparse.SUPPRESS,
                               metavar='INT', help=help_ + ' (e.g. ' + ' '.join(['1000'] * (q.n or 2)) + ')')
        else:
            group.add_argument(*flags, dest=q.name, required=q.required, default=argparse.SUPPRESS,
                               metavar=q.kind.upper() if q.kind != 'value' else 'VALUE', help=help_)
    if cmd.has_kwargs:
        p.add_argument('--kw', action='append', default=[], metavar='KEY=VALUE',
                       help=f'extra keyword argument: {cmd.kwargs_help}')
    _add_global(p)
    p.set_defaults(_command=cmd.name)
    return p


def _add_pipeline_options(p):
    p.add_argument('--workdir', metavar='DIR', help='working directory, overrides [pipeline] workdir')
    p.add_argument('--var', action='append', metavar='NAME=VALUE', help='set a variable used as ${NAME} (repeatable)')


def _build_parser(with_commands=True):
    parser = argparse.ArgumentParser(prog='moraine', description=__doc__.split('\n\n')[0],
                                     epilog='Run `moraine list` for all processing commands.')
    sub = parser.add_subparsers(dest='_sub', metavar='COMMAND')
    p = sub.add_parser('list', help='list the processing commands'); _add_global(p)
    p = sub.add_parser('info', help='shape, dtype and chunks of zarr arrays and pyramids')
    p.add_argument('paths', nargs='+'); _add_global(p)
    p = sub.add_parser('quicklook', help='save a PNG of a pyramid (made by ras-pyramid / pc-pyramid)')
    p.add_argument('pyramid', help='pyramid directory made by `moraine ras-pyramid` or `moraine pc-pyramid`')
    p.add_argument('-o', '--out', help='output PNG, a name ending with .png; .png is appended to a name without a '
                                       'suffix (default: <pyramid name>.png)')
    p.add_argument('--index', type=int, nargs='+', default=[], metavar='I',
                   help='image of a stack, i j of 4D data (..., n, n), or two images (ref sec) with --show intf_all / '
                        'coh / coh_abs (default: the first)')
    p.add_argument('--show', dest='show', choices=['phase', 'intf_0', 'intf_seq', 'intf_all', 'coh', 'coh_abs'],
                   help='what to show of a stack, see moraine.cli.view (default: the phase for complex data, the '
                        'values otherwise)')
    p.add_argument('--width', type=int, default=1000, help='image width in pixels (default: 1000)')
    p.add_argument('--extent', help='part to draw, finer levels for smaller parts: "west,south,east,north" degrees '
                   '(web mercator) or "range_min,azimuth_min,range_max,azimuth_max" pixels (radar grid); '
                   'default: the whole scene')
    p.add_argument('--clim', metavar='LOW,HIGH',
                   help='colour range, the values of the first and the last colour, LOW < HIGH, e.g. --clim -11,6 '
                        'for differences of a few millimetres (default: (-pi, pi) for phases, the 1 - 99 %% range of '
                        'the data otherwise, +-(p99 - p01) for the differences of two images of a real stack)')
    p.add_argument('--image_pairs', metavar='FILE',
                   help='image pairs of the compressed coherence shown with --show coh / coh_abs, a text file of two '
                        'columns, reference and secondary image index, as made by `moraine image-pairs` (default: all '
                        'pairs of the images)')
    _add_global(p)
    p = sub.add_parser('view', help='write a notebook with interactive maps of pyramids (moraine.cli.view)')
    p.add_argument('pyramids', nargs='+', help='pyramid directories made by `moraine ras-pyramid` / `moraine pc-pyramid`')
    p.add_argument('-o', '--out', default='view.ipynb',
                   help='notebook to write, a name ending with .ipynb; .ipynb is appended to a name without a suffix '
                        '(default: view.ipynb)')
    p.add_argument('--show', dest='show', choices=['phase', 'intf_0', 'intf_seq', 'intf_all', 'coh', 'coh_abs'],
                   help='what to show of stacks, see moraine.cli.view (default: the phase for complex data, the values '
                        'otherwise)')
    p.add_argument('--dates', help='toml file with the image dates, e.g. the metadata of load-gamma-metadata')
    p.add_argument('--terrain', action='store_true',
                   help='3D maps over the terrain (public elevation tiles); web mercator pyramids only')
    p.add_argument('--overwrite', action='store_true', help='replace an existing notebook')
    p.add_argument('--image_pairs', metavar='FILE',
                   help='image pairs of the compressed coherence shown with --show coh / coh_abs, a text file of two '
                        'columns, reference and secondary image index, as made by `moraine image-pairs` (default: all '
                        'pairs of the images)')
    _add_global(p)
    p = sub.add_parser('run', help='run or resume a pipeline file (TOML)')
    p.add_argument('pipeline')
    p.add_argument('--dry-run', action='store_true', help='check the file and print the plan only')
    p.add_argument('--from', dest='from_step', metavar='STEP', help='rerun from this step on')
    p.add_argument('--only', metavar='STEP', action='append', help='run only this step (repeatable)')
    p.add_argument('--force', action='store_true', help='rerun steps that are already done')
    p.add_argument('--no-quicklook', action='store_true', help='do not save quicklook images')
    _add_pipeline_options(p)
    _add_global(p)
    p = sub.add_parser('status', help='show the state of a pipeline')
    p.add_argument('pipeline'); _add_pipeline_options(p); _add_global(p)
    if with_commands:
        for cmd in commands().values():
            _add_command_parser(sub, cmd)
    return parser


# ---------------------------------------------------------------- entry point

def _run(args):
    sub = args._sub
    if sub == 'list':
        cmds = commands()
        result = {'commands': [{'name': c.name, 'module': c.module, 'summary': c.summary} for c in cmds.values()]}
        def text():
            for mod in _MODULES:
                group = [c for c in cmds.values() if c.module == mod]
                if group:
                    print(f'{mod}:')
                    for c in group:
                        print(f'  {c.name:32s} {c.summary.split(". ")[0][:90]}')
            print('\nOther: info, quicklook, view, run, status.  `moraine COMMAND --help` for details.')
        return _emit(args, result, text)
    if sub == 'info':
        from .summary import summarize, _pyramid_input
        summaries = []
        for path in args.paths:
            _pyramid_input(path, pyramid=False)     # a pyramid to rebuild: a usage error, as for quicklook and view
            summaries.append(summarize(path))
        result = {'summaries': summaries}
        return _emit(args, result, lambda: [_print_summary(s) for s in result['summaries']])
    if sub == 'quicklook':
        from .summary import quicklook
        out = quicklook(args.pyramid, args.out or Path(args.pyramid.rstrip('/')).name + '.png', index=tuple(args.index),
                        show=args.show, width=args.width, extent=args.extent, clim=args.clim,
                        image_pairs=args.image_pairs)
        return _emit(args, {'png': out}, lambda: print(f'saved {out}'))
    if sub == 'view':
        from .summary import view
        out = view(args.pyramids, args.out, show=args.show, dates=args.dates, terrain=args.terrain,
                   overwrite=args.overwrite, image_pairs=args.image_pairs)
        return _emit(args, {'notebook': out, 'pyramids': args.pyramids},
                     lambda: print(f'saved {out}: open it in Jupyter or VS Code and run all cells'))
    if sub in ('run', 'status'):
        from . import pipeline
        return pipeline.cli(args, _emit)
    cmd = get_command(args._command)
    values = {q.name: getattr(args, q.name) for q in cmd.params if hasattr(args, q.name)}
    kw = {}
    for item in getattr(args, 'kw', []) or []:
        key, sep, val = item.partition('=')
        if not sep:
            raise UsageError(f'--kw expects KEY=VALUE, got {item!r}')
        kw[key] = val
    record = execute(cmd, bind_args(cmd, values, kw))
    record['ok'] = True
    def text():
        print(f'{cmd.name} finished in {record["seconds"]} s')
        for s in record['summaries']:
            _print_summary(s)
    return _emit(args, record, text)


# options of quicklook whose value is a list of numbers that may start with a minus sign
_NUMBER_OPTIONS = ('--clim', '--extent')


def _normalize_options(argv):
    """Accept --is-shp-dir for --is_shp_dir (and --no-x-y for --no-x_y) in processing commands, quicklook and view,
    and a value of `_NUMBER_OPTIONS` of quicklook that starts with a minus sign (--clim -11,6 for --clim=-11,6,
    --extent -122.5,37.5,-122,38 for --extent=-122.5,37.5,-122,38)."""
    if not argv or argv[0] in ('list', 'info', 'run', 'status'):
        return argv
    out = []
    for tok in argv:
        # argparse takes '-11,6' for an unknown option: only a single number may start with a minus sign
        if argv[0] == 'quicklook' and out and out[-1] in _NUMBER_OPTIONS and re.match(r'-\.?\d', tok):
            out[-1] = f'{out[-1]}={tok}'
            continue
        if tok.startswith('--') and len(tok) > 2:
            name, eq, value = tok[2:].partition('=')
            prefix = 'no-' if name.startswith('no-') else ''
            name = prefix + name[len(prefix):].replace('-', '_')
            tok = f'--{name}{eq}{value}'
        out.append(tok)
    return out


def main(argv=None):
    """Entry point of the ``moraine`` command."""
    argv = _normalize_options(sys.argv[1:] if argv is None else list(argv))
    # the processing commands import the whole library; skip that for --help of the top level
    needs_commands = not argv or argv[0] not in ('info', 'quicklook', 'view', 'status') or '--help' in argv
    parser = _build_parser(with_commands=needs_commands or argv[0] == 'run')
    args = parser.parse_args(argv)
    if not args._sub:
        parser.print_help(); return 0
    _setup_logging(args)
    try:
        return _run(args) or 0
    except Exception as e:
        usage = isinstance(e, (UsageError, ArgumentError))     # also arguments that a command rejects
        if getattr(args, 'traceback', False):
            import traceback; traceback.print_exc()
        msg = str(e) if usage else f'{type(e).__name__}: {e}'
        if getattr(args, 'json', False):
            print(json.dumps({'version': JSON_VERSION, 'ok': False, 'error': msg}))
        print(f'moraine: error: {msg}', file=sys.stderr)
        if not getattr(args, 'traceback', False) and not usage:
            print('(add --traceback for details)', file=sys.stderr)
        return 2 if usage else 1
