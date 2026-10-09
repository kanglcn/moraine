# Contract: pipeline files

Version 1. TOML files run by `moraine run FILE` and inspected by `moraine status FILE` (decision 0006).

## Tables

| table / key | required | description |
|---|---|---|
| `[pipeline]` | no | settings of the file |
| `[pipeline] version` | no | format version, integer, default 1; newer versions than the installed moraine supports are rejected |
| `[pipeline] workdir` | no | working directory, relative to the file; default: the directory of the file; `--workdir` overrides it |
| `[pipeline] quicklook` | no | save PNGs of the pyramids made by the steps, default true |
| `[vars]` | no | variables; `${name}` is replaced in every value of `[defaults]` and `[[step]]`; `--var name=value` overrides |
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
- `image_pairs` is a file with two integer columns (made by the `image-pairs` command) or a list of pairs;
  a file is read when the step runs, so an earlier step may make it.
- An unknown `${name}` is an error.

## Execution

- Steps run in file order; the first failing step stops the run (exit code 1).
- A step is skipped when it finished before with the same command and arguments (hash), all its outputs
  exist and none of its inputs changed (modification time). Otherwise it runs; steps reading the outputs
  of a step that runs are rerun too. `--force`, `--from STEP` and `--only STEP` override this.
- `--dry-run` prints the plan with the reason for each step and runs nothing.
- Outputs and inputs of a step are the argument paths it created / modified and read.

## State

Kept in `<workdir>/.moraine/<file name without .toml>/`: `state.json` (per step record), `logs/<step>.log`,
`quicklook/<step>__<output>.png`. The layout of `state.json` is internal and may change; use
`moraine status FILE --json` (json-output contract) to read the state.
