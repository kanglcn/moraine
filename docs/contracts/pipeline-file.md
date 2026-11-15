# Contract: pipeline files

Version 1. TOML files run by `moraine run FILE` and inspected by `moraine status FILE` (decision 0006).

## Tables

| table / key | required | description |
|---|---|---|
| `[pipeline]` | no | settings of the file |
| `[pipeline] version` | no | format version, integer, default 1; newer versions than the installed moraine supports are rejected |
| `[pipeline] workdir` | no | working directory, relative to the file; default: the directory of the file; `--workdir` overrides it |
| `[pipeline] quicklook` | no | save PNGs of the pyramids made by the steps, default true |
| `[vars]` | no | variables, TOML values; `${name}` is replaced in every value of `[defaults]` and `[[step]]`; `--var name=value` overrides (the value is text) |
| `[vars] NAME = { file = "PATH", key = "KEY" }` | no | the variable NAME is the value of the top-level key KEY of the TOML file PATH, relative to the working directory |
| `[defaults]` | no | arguments given to every step whose command has them |
| `[[step]]` | yes | the steps, run in file order |
| `name` | yes | unique step name |
| `run` | yes | a command of `moraine list` |
| `quicklook` | no | per step override of `[pipeline] quicklook` |
| `kw` | no | table of extra keyword arguments passed to the function (`**kwargs`), e.g. the input arrays of `math` |
| any other key | - | an argument of the command; unknown names are errors |

No other top level tables and no other `[pipeline]` keys are allowed.

## Values

- Paths are relative to the working directory; each step runs with the working directory as current
  directory.
- Booleans, integers, floats and strings are TOML values. Tuples (`tuple[int, int]` arguments) are TOML
  lists `[1000, 1000]` or strings `"1000,1000"`; their number of integers is checked.
- Arguments accepting several paths take a TOML list.
- `image_pairs` is a file with two integer columns (made by the `image-pairs` command) or a list of pairs,
  sorted without duplicates and with every reference index smaller than its secondary index
  ([data conventions](data.md)); a file is read when the step runs, so an earlier step may make it.
- A value that is exactly `"${name}"` is the value of the variable, with its type: a list stays a list, a number
  a number. Inside a longer string `${name}` is replaced by the text of the value; a list or a table there is an
  error. An unknown `${name}` is an error.
- A variable `{ file = "PATH", key = "KEY" }` is read when the pipeline file is loaded and again before each step
  using it runs, so an earlier step or pipeline may make the file. While the file does not exist, the steps
  using the variable are planned to run (`run (variable NAME: PATH does not exist yet)`) and fail if it still
  does not exist when they run. A file that is not TOML or has no key KEY is an error. Variables are not
  replaced inside `[vars]`.

## Execution

- Steps run in file order; the first failing step stops the run (exit code 1).
- A step is skipped when it finished before with the same command and arguments (hash), all its outputs
  exist and none of its inputs changed (modification time). Otherwise it runs; steps reading the outputs
  of a step that runs are rerun too. `--force`, `--from STEP` and `--only STEP` override this.
- The hash of a step holds the values of the variables it reads from files: the step reruns when such a value
  changes, not when only other keys of the file change. The file of a variable counts as read by the steps
  using it, so they are rerun with the step that makes it.
- `--dry-run` prints the plan with the reason for each step and runs nothing.
- Outputs and inputs of a step are the argument paths it created / modified and read.

## State

Kept in `<workdir>/.moraine/<file name without .toml>/`: `state.json` (per step record), `logs/<step>.log`,
`quicklook/<step>__<output>.png`. The layout of `state.json` is internal and may change; use
`moraine status FILE --json` (json-output contract) to read the state.
