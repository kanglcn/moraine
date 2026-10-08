"""mkdocs-gen-files entry: write the generated reference pages (docs/_scripts/refgen.py, decision 0032)."""
import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).resolve().parent / '_scripts'))
import refgen  # noqa: E402

for page, (text, edit) in refgen.all_pages().items():
    with mkdocs_gen_files.open(page, 'w') as f:
        f.write(text)
    if edit:
        mkdocs_gen_files.set_edit_path(page, edit)
