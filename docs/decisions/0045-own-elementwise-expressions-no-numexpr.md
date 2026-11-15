# 0045 Elementwise expressions are evaluated by moraine; numexpr and pandas are not dependencies

## Status

Accepted

## Date

2026-11-14

## Context

The `operation` of `math`, `pc-logic-ras` and `pc-logic-pc` (thresholds and masks such as `ras<0.4`,
`(pc_in>=0.85)&(pc_in<=1.0)` or `(lon > 14.225) & (lat > 40.82)`) was evaluated by numexpr. numexpr starts its pool of
threads when it is imported: as many as `NUMEXPR_NUM_THREADS` or `OMP_NUM_THREADS` ask for (128 on the development
cluster), else up to 16. `moraine/cli/__init__.py` imports every command module, so every moraine process carried these
idle threads, every later fork (the pyramids) was a fork of a multi-threaded process with Python's warning, and `math`
ran numexpr's threads inside the threads of the executor. The three commands need elementwise arithmetic, comparisons
and logic on chunks of a few million values, where numexpr's speed on one large array in many threads does not matter
next to the zarr I/O; its reductions (`sum`, `min`, ...) gave a result per chunk in `math`. pandas, which moraine
used for a list of files and one column of a GAMMA text file in `load.py`, imports numexpr wherever it is installed.

## Decision

- `moraine.api.utils_.expr.evaluate(expression, arrays)` evaluates the expressions. The expression is parsed with
  Python's `ast`; only numbers, the names of the arrays, parentheses, the operators and comparisons of numexpr
  (`+ - * / ** % << >>`, `< <= > >= == !=`, `& | ~`) and numexpr's elementwise functions are allowed, computed with
  numpy. Nothing is passed to `eval`.
- The dtype follows numpy's rules: a Python number does not change the dtype of an array (float32 times 2.5 is
  float32, numexpr gave float64), `abs` of complex data is real. A division by zero or an invalid value gives inf or
  nan without a warning, as with numexpr.
- Not supported: the reductions `sum`, `prod`, `min`, `max` and the string function `contains`. `and`, `or`, `not`,
  chained comparisons, `//` and `^` are errors, as in numexpr.
- numexpr and pandas are not dependencies of moraine: `load.py` keeps the files of a GAMMA directory in lists and
  reads the baselines of `base_calc` with `np.loadtxt`.

## Consequences

- Comparisons, logic, integer arithmetic and `where` give the same values bit for bit (all expressions of the
  examples, tests and guides); transcendental functions differ in the last bit (relative 3.5e-7 in float32, 4e-16 in
  float64). `math` of float32 data with a float constant writes float32 instead of float64.
- On one thread and a chunk of 1000 x 1000 float32 values, a comparison takes 0.8 instead of 3.2 ms and
  `sin(a)*exp(b)/2` 9.9 instead of 4.7 ms; the executor runs the chunks in parallel. The commands take about the
  same time and memory: on 100 million float64 pixels `math` 2.0 instead of 2.4 s (a comparison) and 4.1 instead of
  5.5 - 7.0 s (sines and exponentials), `pc-logic-ras` 1.2 instead of 1.1 s.
- `import moraine, moraine.cli, moraine.command` imports neither numexpr nor pandas and starts no thread (130 before
  on the development cluster), also where numexpr is still installed (`tests/test_architecture.py`).

## Do not

- Do not evaluate the expressions of pipeline files or command lines with `eval` or `exec`.
- Do not use numexpr in moraine, nor pandas, which imports it.
