"""Generate the reference pages of the manual from the code (decision 0032).

``cli_pages()`` writes one page per ``moraine.cli`` module from the command registry
(``moraine.command.commands()``, the source of ``moraine COMMAND --help``) with the steps of ``examples/*.toml``
that use each command; ``builtin_page()`` captures the ``--help`` of the built-in commands; ``api_pages()`` writes one
mkdocstrings page per ``moraine.api`` module with its ``__all__`` names and the examples of ``docs/api/examples/``;
``tutorial_pages()`` converts the notebooks of ``nbs/Tutorials/`` to markdown (text and code cells, no outputs).

Every function returns ``{page path relative to docs/: (markdown text, edit path or None)}``. ``docs/gen_ref_pages.py``
writes them through mkdocs-gen-files; ``tests/test_docs.py`` checks them without building the site.
"""

import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / 'docs'
EXAMPLES = sorted((REPO / 'examples').glob('*.toml'))
GITHUB = 'https://github.com/kanglcn/moraine/blob/main'

# moraine.api modules documented, in the order of the API index; a tuple lists the modules of a subpackage
API_MODULES = ['calamp', 'pc', 'rtree', 'tnet', 'polygon', 'ps', 'shp', 'co', 'pl', 'dl', 'pqm',
               ('unwrap', ['delaunay_', 'mcf', 'emcf', 'closure', 'gamma']), 'utils_']
BUILTIN = ['list', 'info', 'quicklook', 'view', 'run', 'status']
TUTORIALS = ['CampiFlegrei/01_load', 'CampiFlegrei/02_ps', 'CampiFlegrei/03_ds', 'CampiFlegrei/04_refine',
             'CampiFlegrei/05_unwrap', 'Xinpu/01_load', 'Xinpu/02_ps', 'Xinpu/03_ds', 'Xinpu/04_refine', 'Xinpu/05_unwrap',
             'Adaptive_Multilook', 'DS_Processing']


def _module_summary(module) -> str:
    """First line of a module docstring, as a sentence."""
    doc = (inspect.getdoc(module) or '').strip().splitlines()
    return doc[0].strip().rstrip('.') if doc else ''


# ---------------------------------------------------------------- command line

def _example_steps() -> dict:
    """{command name: [(example file name, step name, TOML text of the step), ...]} from examples/*.toml."""
    out = {}
    for path in EXAMPLES:
        blocks = path.read_text().split('\n[[step]]')
        for block in blocks[1:]:
            text = '[[step]]' + block.rstrip() + '\n'
            # the step ends where the next top level table starts (none in the examples)
            run = re.search(r'^run\s*=\s*"([^"]+)"', text, re.M)
            name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
            if run and name:
                out.setdefault(run.group(1), []).append((path.name, name.group(1), text))
    return out


def _usage(cmd) -> str:
    """One line like `moraine amp-disp --rslc STR --adi STR [--chunks INT INT] [--cuda]`."""
    parts = [f'moraine {cmd.name}']
    for p in cmd.params:
        if p.kind == 'bool':
            tok = f'--{p.name} | --no-{p.name}'
        elif p.kind == 'list':
            tok = f'--{p.name} PATH [PATH ...]'
        elif p.kind == 'tuple':
            tok = f'--{p.name} ' + ' '.join(['INT'] * (p.n or 2))
        elif p.kind == 'pairs':
            tok = f'--{p.name} FILE'
        else:
            tok = f'--{p.name} {"STR" if p.kind == "str" else "VALUE"}'
        parts.append(tok if p.required else f'[{tok}]')
    if cmd.has_kwargs:
        parts.append('[--kw KEY=VALUE]')
    return ' '.join(parts)


def _io_badge(desc: str) -> str:
    m = re.match(r'\s*(input|output)\s*:\s*', desc, re.I)
    if not m:
        return desc
    kind = m.group(1).lower()
    return f'<span class="mo-io mo-io--{"in" if kind == "input" else "out"}">{kind}</span> ' + desc[m.end():]


def _cell(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' ').strip()


def _type_and_default(p) -> tuple:
    """Type column and default column of a parameter from the docstring type and the real default."""
    typ = p.type_doc or ''
    typ = re.sub(r',?\s*(optional|default\s*:\s*[^,]+)', '', typ).strip(' ,')
    if p.required:
        return typ, 'required'
    return typ, f'`{p.default!r}`'


def _api_twin(cmd) -> str:
    """Markdown link to the API function of the same name, if there is one."""
    import moraine
    f = getattr(moraine, cmd.func.__name__, None)
    if f is None or not callable(f) or getattr(f, '__module__', '') is None:
        return ''
    if not f.__module__.startswith('moraine.api'):
        return ''
    return f'[`mr.{f.__name__}`][{f.__module__}.{f.__name__}]'


def _python_call(cmd) -> str:
    raw = getattr(cmd.func, '__wrapped__', cmd.func)
    sig = inspect.signature(raw)
    args = []
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            args.append(f'**{p.name}')
        elif p.default is inspect.Parameter.empty:
            args.append(p.name)
        else:
            args.append(f'{p.name}={p.default!r}')
    return f'mc.{raw.__name__}({", ".join(args)})'


def _command_section(cmd, examples: dict) -> str:
    lines = [f'## {cmd.name}', '', cmd.summary, '', '```bash', _usage(cmd), '```', '']
    lines += ['| argument | type | default | description |', '|---|---|---|---|']
    for p in cmd.params:
        typ, default = _type_and_default(p)
        lines.append(f'| `--{p.name}` | {_cell(typ)} | {default} | {_cell(_io_badge(p.help))} |')
    if cmd.has_kwargs:
        lines.append(f'| `--kw KEY=VALUE` | | | extra keyword argument, repeatable: {_cell(cmd.kwargs_help)} |')
    lines.append('')
    lines.append('Global options: `--json`, `--traceback`, `--log FILE`, `-q` / `--quiet` ([conventions](index.md)).')
    lines.append('')
    used = examples.get(cmd.name, [])
    if used:
        file, step, text = used[0]
        lines += [f'**In a pipeline** (`examples/{file}`, step `{step}`):', '', '```toml', text.rstrip(), '```', '']
        others = [(f, s) for f, s, _ in used[1:]]
        if others:
            lines.append('Also in ' + ', '.join(f'`examples/{f}` (step `{s}`)' for f, s in others) + '.')
            lines.append('')
    else:
        req = [p for p in cmd.params if p.required]
        skel = ['[[step]]', f'name = "{cmd.name.replace("-", "_")}"', f'run = "{cmd.name}"']
        skel += [f'{p.name} = "..."' for p in req]
        lines += ['**In a pipeline** (not used by the example pipelines; required arguments only):', '',
                  '```toml', *skel, '```', '']
    twin = _api_twin(cmd)
    lines += ['**In Python**:', '', '```python', 'import moraine.cli as mc', _python_call(cmd), '```', '']
    if twin:
        lines.append(f'The array function {twin} does the same on numpy / cupy arrays in memory.')
        lines.append('')
    return '\n'.join(lines)


def cli_pages() -> dict:
    from moraine.command import commands
    cmds = commands()
    examples = _example_steps()
    modules = []
    for c in cmds.values():
        if c.module not in modules:
            modules.append(c.module)
    pages = {}
    for mod in modules:
        m = importlib.import_module(f'moraine.cli.{mod}')
        group = [c for c in cmds.values() if c.module == mod]
        summary = _module_summary(m)
        head = [f'# `{mod}`: {summary}' if summary else f'# `{mod}`', '',
                f'Commands of `moraine.cli.{mod}` ([source]({GITHUB}/moraine/cli/{mod}.py)); the same functions are called in '
                f'Python as `mc.<name>(...)` with `import moraine.cli as mc`. The help below is generated from the '
                f'docstrings, like `moraine COMMAND --help`.', '']
        head += ['| command | summary |', '|---|---|']
        head += [f'| [`{c.name}`](#{c.name}) | {_cell(c.summary.split(". ")[0])} |' for c in group]
        head.append('')
        body = [_command_section(c, examples) for c in group]
        pages[f'cli/{mod}.md'] = ('\n'.join(head) + '\n' + '\n'.join(body), f'../moraine/cli/{mod}.py')
    return pages


def _help(args) -> str:
    env = dict(os.environ, COLUMNS='96', PYTHONWARNINGS='ignore')
    out = subprocess.run([sys.executable, '-m', 'moraine', *args, '--help'], capture_output=True, text=True,
                         env=env, cwd=REPO)
    return out.stdout.strip()


def builtin_page() -> dict:
    """The built-in commands (not generated from moraine.cli) with their --help, and `moraine list`."""
    lines = ['# Built-in commands', '',
             'The commands that inspect results, draw pictures, write notebooks and run pipeline files. Their help is '
             'captured from `moraine COMMAND --help`; what to do with them is on [Checking results](../start/checking.md) '
             'and [Workflows](../workflows/index.md).', '']
    for name in BUILTIN:
        lines += [f'## {name}', '', '```text', _help([name]), '```', '']
    env = dict(os.environ, COLUMNS='120', PYTHONWARNINGS='ignore')
    listing = subprocess.run([sys.executable, '-m', 'moraine', 'list'], capture_output=True, text=True, env=env,
                             cwd=REPO).stdout.strip()
    lines += ['## The processing commands', '', 'The output of `moraine list`:', '', '```text', listing, '```', '']
    return {'cli/builtin.md': ('\n'.join(lines), None)}


# ---------------------------------------------------------------- python api

def _api_module_page(name, submodules=None) -> str:
    pkg = f'moraine.api.{name}'
    m = importlib.import_module(pkg)
    summary = _module_summary(m)
    title = name.rstrip('_')
    src = f'moraine/api/{name}/__init__.py' if submodules else f'moraine/api/{name}.py'
    lines = [f'# `{title}`: {summary}' if summary else f'# `{title}`', '', f'`{pkg}` ([source]({GITHUB}/{src})). ']
    members = []
    lines.append('')

    def directive(path, level):      # one mkdocstrings block per public name: signature, docstring, no module text
        return [f'::: {path}', '    options:', '      show_root_heading: true', '      show_root_toc_entry: true',
                f'      heading_level: {level}', '']

    if submodules:
        for sub in submodules:
            sm = importlib.import_module(f'{pkg}.{sub}')
            names = list(getattr(sm, '__all__', []))
            members += names
            sub_summary = _module_summary(sm)
            lines += [f'## {sub.rstrip("_")}' + (f': {sub_summary}' if sub_summary else ''), '']
            for n in names:
                lines += directive(f'{pkg}.{sub}.{n}', 3)
    else:
        names = list(getattr(m, '__all__', []))
        members += names
        for n in names:
            lines += directive(f'{pkg}.{n}', 2)
    import moraine
    missing = [n for n in members if not hasattr(moraine, n)]
    if not missing:
        lines[2] += 'Every name is also `moraine.<name>` (`import moraine as mr`).'
    elif len(missing) == len(members):
        lines[2] += f'These names are not re-exported: import them from `{pkg}`' + ('.' if not submodules else ' and its modules.')
    else:
        lines[2] += ('Every name is also `moraine.<name>` (`import moraine as mr`) except '
                     + ', '.join(f'`{n}`' for n in missing) + ', imported from the module.')
    example = DOCS / 'api' / 'examples' / f'{title}.md'
    if example.exists():
        lines += ['## Examples', '', 'Run when this manual is built; the output follows each snippet.', '',
                  f'--8<-- "api/examples/{title}.md"', '']
    return '\n'.join(lines), members


def api_pages() -> dict:
    pages = {}
    for item in API_MODULES:
        name, subs = item if isinstance(item, tuple) else (item, None)
        text, _ = _api_module_page(name, subs)
        src = f'../moraine/api/{name}/__init__.py' if subs else f'../moraine/api/{name}.py'
        pages[f'api/{name.rstrip("_")}.md'] = (text, src)
    return pages


def api_members() -> dict:
    """{page path: [documented names]}, for the tests."""
    out = {}
    for item in API_MODULES:
        name, subs = item if isinstance(item, tuple) else (item, None)
        out[f'api/{name.rstrip("_")}.md'] = _api_module_page(name, subs)[1]
    return out


# ---------------------------------------------------------------- tutorials

_CALLOUT = re.compile(r'^:::\s*\{\.callout-(\w+)\}\s*\n(.*?)^:::\s*$', re.S | re.M)


def _markdown_cell(src: str) -> str:
    src = _CALLOUT.sub(lambda m: f'!!! {m.group(1)}\n' + ''.join(f'    {l}\n' for l in m.group(2).strip().splitlines()), src)
    src = re.sub(r'\]\(([^)]+?)\.ipynb\)', r'](\1.md)', src)      # links between notebooks
    return src.strip()


def notebook_to_markdown(path: Path, rel: str) -> str:
    nb = json.loads(path.read_text())
    out = []
    for cell in nb.get('cells', []):
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue
        if cell['cell_type'] == 'markdown':
            out.append(_markdown_cell(src))
        elif cell['cell_type'] == 'code':
            out.append('```python\n' + src.rstrip() + '\n```')
    text = '\n\n'.join(out) + '\n'
    note = (f'!!! info "Notebook"\n    Converted from [`{rel}`]({GITHUB}/{rel}) (text and code cells; the outputs are not '
            'stored in the repository). Run it in Jupyter or VS Code for the interactive maps.\n\n')
    # the first heading stays the title; the note goes after it
    m = re.match(r'(# .*?\n)', text)
    return text[:m.end()] + '\n' + note + text[m.end():].lstrip('\n') if m else note + text


def tutorial_pages() -> dict:
    pages = {}
    for t in TUTORIALS:
        rel = f'nbs/Tutorials/CLI/{t}.ipynb' if '/' in t else f'nbs/Tutorials/{t}.ipynb'
        path = REPO / rel
        if not path.exists():
            continue
        pages[f'tutorials/{t}.md'] = (notebook_to_markdown(path, rel), f'../{rel}')
    return pages


def all_pages() -> dict:
    pages = {}
    for part in (cli_pages(), builtin_page(), api_pages(), tutorial_pages()):
        pages.update(part)
    return pages
