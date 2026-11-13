// MathJax for the formulas of the manual (decision 0042). pymdownx.arithmatex (generic mode) wraps every formula
// in an element of class "arithmatex" with \( \) or \[ \] delimiters; only these elements are typeset.
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: '.*|',
    processHtmlClass: 'arithmatex',
  },
};

// typeset again when the theme replaces the content of the page without a reload
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
