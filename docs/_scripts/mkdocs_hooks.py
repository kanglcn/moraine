"""MkDocs hooks of the manual (decision 0037).

Before the files are collected, the generated reference pages of ``refgen.py`` are written into ``docs/`` (they are
ignored by git), so that every plugin, the i18n plugin included, sees them as ordinary pages; their "edit" links
point to the source they are generated from.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refgen  # noqa: E402

_EDIT = {}


def on_config(config):
    # only the API objects are cross-reference targets, not every heading of the manual: the type `list` in a
    # signature must not link to the heading "list" of the built-in commands page
    autorefs = config['plugins'].get('autorefs')
    if autorefs is not None:
        autorefs.scan_toc = False
    docs = Path(config['docs_dir'])
    pages = refgen.all_pages()
    _remove_stale_pages(docs, pages)
    for page, (text, edit) in pages.items():
        path = docs / page
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != text:    # unchanged files are not rewritten (mkdocs serve)
            path.write_text(text)
        if edit:
            _EDIT[page] = edit
    return config


def _remove_stale_pages(docs, pages):
    """Remove the pages of an earlier build that are not generated any more (a module or a tutorial was removed or
    renamed): like the generated pages they are ignored by git, and they would be built with their old content."""
    dirs = {(docs / page).parent for page in pages}
    old = [str(p) for d in dirs if d.is_dir() for p in d.glob('*.md') if p.relative_to(docs).as_posix() not in pages]
    if not old:
        return
    try:
        out = subprocess.run(['git', 'check-ignore', '--', *old], cwd=docs, capture_output=True, text=True)
    except OSError:      # no git: nothing is removed
        return
    for path in out.stdout.splitlines():
        Path(path).unlink(missing_ok=True)


def on_page_context(context, page, config, nav):
    src = page.file.src_uri
    edit = _EDIT.get(src) or _EDIT.get(src.replace('.zh.md', '.md'))
    if edit and config.get('repo_url'):
        page.edit_url = config['repo_url'].rstrip('/') + '/blob/main/' + edit.removeprefix('../')
    elif src in refgen.all_pages() or src.replace('.zh.md', '.md') in refgen.all_pages():
        page.edit_url = None
    return context
