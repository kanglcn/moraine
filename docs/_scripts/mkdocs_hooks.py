"""MkDocs hooks of the manual (decision 0037).

Before the files are collected, the generated reference pages of ``refgen.py`` are written into ``docs/`` (they are
ignored by git), so that every plugin, the i18n plugin included, sees them as ordinary pages; their "edit" links
point to the source they are generated from.
"""
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
    for page, (text, edit) in refgen.all_pages().items():
        path = docs / page
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != text:    # unchanged files are not rewritten (mkdocs serve)
            path.write_text(text)
        if edit:
            _EDIT[page] = edit
    return config


def on_page_context(context, page, config, nav):
    src = page.file.src_uri
    edit = _EDIT.get(src) or _EDIT.get(src.replace('.zh.md', '.md'))
    if edit and config.get('repo_url'):
        page.edit_url = config['repo_url'].rstrip('/') + '/blob/main/' + edit.removeprefix('../')
    elif src in refgen.all_pages() or src.replace('.zh.md', '.md') in refgen.all_pages():
        page.edit_url = None
    return context
