# 0042 Formulas in the manual are LaTeX, typeset by MathJax

## Status

Accepted

## Date

2026-11-12

## Context

The manual (decision 0037) explains methods whose reasoning is mathematical, first the principle of EMCF
unwrapping; formulas written as plain text are hard to read and to check. The markdown of `docs/` had no support
for formulas. Options: MathJax or KaTeX in the browser, with `pymdownx.arithmatex` marking the formulas (part of
pymdown-extensions, which MkDocs Material already requires), or formulas rendered at build time (another MkDocs
plugin with a JavaScript runtime or a port of KaTeX in the build).

## Decision

- Formulas in the pages of `docs/` are LaTeX between `$...$` (inline) or `$$...$$` (display), marked by
  `pymdownx.arithmatex` in generic mode.
- MathJax 3 (`tex-mml-chtml.js` of a fixed version from the jsdelivr CDN) typesets them in the browser;
  `docs/assets/js/mathjax.js` makes it read only the marked formulas and typeset again when the theme replaces
  the content of a page.
- Docstrings keep plain text formulas: they are also the `--help` text of the commands (decision 0005).

## Consequences

- No new Python dependency. The browser needs internet access to the CDN to show typeset formulas; without it
  the LaTeX source is shown. MathJax is loaded on every page and typesets only the marked formulas.
- `mkdocs build --strict` does not check the LaTeX: look at the page after a change of a formula.
- A `$` in the text of a page or a docstring (not in code) can start a formula: write it in code or escape it.

## Do not

- Do not put LaTeX into docstrings or the `--help` text.
- Do not load MathJax from a URL without a fixed version.
