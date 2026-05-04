# dace_deterministic_codegen

Determinism check for gt4py's DaCe backend. Runs an icon4py test selection
through `nox` **twice** with isolated gt4py build caches, then compares
the generated source code under each program's `src/` between the two
runs. Exit 0 = identical (deterministic), exit 1 = different.

Currently supports the **cpu**, **cuda**, and **HIP** dace backends.
HIP is supported transparently: dace emits HIP code under `src/cuda/hip/`
(target_name="cuda", target_type="hip"), and the harness's recursive
sweep of `src/cuda/` picks it up automatically. If a run emits anything
else under `src/` (mpi, sve, mlir, snitch, …) the harness fails
immediately with a clear message — silently ignoring an unfamiliar
backend would mean reporting "deterministic" without actually checking
the relevant code.

Valid `--selection` and `--component` values are read from icon4py's
own `noxfile.py` at runtime — no hardcoding here, so the harness
auto-tracks any future changes to icon4py's parametrization.

Mirrors icon4py's `ci/dace.yml`, with the session name configurable:

```bash
nox -r -s "<session>-<python>(<selection>, <component>)" -- <posargs>
```

Default `<session>` is `test_model` — what `ci/dace.yml` itself uses.

## A note on paths

Every `--*` flag that takes a path (`--icon4py`, `--gt4py`, `--dace`)
accepts **both absolute and relative** paths. Relative paths are
resolved against the current working directory — i.e. wherever you
invoke the script from, not where the script lives. The script prints
the resolved absolute path on startup whenever you pass a relative one,
so you can confirm what it landed on.

## Setup (one-time)

Done once per machine, before any check is run.

**1. Activate the gt4py venv** with editable gt4py (and dace, if on a
custom branch):

```bash
source /path/to/gt4py-venv/bin/activate
uv pip install -e /path/to/gt4py
uv pip install -e /path/to/dace        # optional, if custom dace branch
```

**2. Bootstrap icon4py into that same venv.** This patches icon4py's
`[tool.uv.sources]` so the editable gt4py / dace are what `uv sync`
installs into nox's session venv:

```bash
uv pip install tomli_w
python /path/to/gt4py/ci/dace_deterministic_codegen/bootstrap_icon4py.py \
    --icon4py /path/to/icon4py \
    --gt4py   /path/to/gt4py \
    --dace    /path/to/dace          # omit if upstream dace
```

**3. Sanity check:**

```bash
python -c "import gt4py.next; print(gt4py.next.__file__)"
# must print a path inside your gt4py checkout, NOT site-packages/
```

## Run the check

With the venv from step 1 active:

```bash
python /path/to/gt4py/ci/dace_deterministic_codegen/dace_deterministic_codegen.py \
    --icon4py   /path/to/icon4py \
    --selection <selection> \
    --component <component> \
    --posarg=--backend=dace_cpu \
    --posarg=--grid=icon_regional
```

The valid values for `--selection` and `--component` are read directly
from icon4py's `noxfile.py` at runtime. As of icon4py main, that's:

- `--selection`: `datatest`, `stencils`, `basic`
- `--component`: `advection`, `diffusion`, `dycore`, `microphysics`,
  `muphys`, `common`, `driver`, `standalone_driver`, `testing`

If icon4py adds or renames these, the harness picks it up automatically;
no update needed here. If you pass an invalid value, the error message
lists the actual valid set extracted from your icon4py checkout.

## Examples

**Stencils for muphys, CPU** — mirrors `ci/dace.yml`'s stencil pattern:

```bash
python $GT4PY/ci/dace_deterministic_codegen/dace_deterministic_codegen.py \
    --icon4py $ICON4PY \
    --selection stencils \
    --component muphys \
    --posarg=--backend=dace_cpu \
    --posarg=--grid=icon_regional
```

**Datatest for dycore, GPU** — mirrors the datatest pattern:

```bash
python $GT4PY/ci/dace_deterministic_codegen/dace_deterministic_codegen.py \
    --icon4py $ICON4PY \
    --selection datatest \
    --component dycore \
    --posarg=--backend=dace_gpu \
    --posarg=--level=integration
```

**Custom session** — say a future icon4py defines a `test_other`
session with the same parametrization shape:

```bash
python $GT4PY/ci/dace_deterministic_codegen/dace_deterministic_codegen.py \
    --icon4py $ICON4PY \
    --session test_other \
    --selection stencils \
    --component muphys \
    --posarg=--backend=dace_cpu
```

## Output

By default, everything lands at `<icon4py>/_dace_deterministic_codegen/`.
Override with `--workdir PATH` (absolute or relative):

```
<workdir>/
├── run1/.gt4py_cache/...       run1/test.log
├── run2/.gt4py_cache/...       run2/test.log
├── diffs/<program>/<file>.diff   (only on mismatch)
└── report.txt                    (human-readable summary)
```

**Re-running wipes the workdir.** Whatever was there before — old logs,
old caches, an old `report.txt` from yesterday — is removed before the
new run starts. No merging, no appending. If you want to keep history
across invocations, copy the directory before re-running.

## Exit codes

| Code | Meaning                                                                   |
| ---- | ------------------------------------------------------------------------- |
| 0    | Codegen is deterministic.                                                 |
| 1    | Codegen differs (see `report.txt` and `diffs/`).                          |
| 2    | Bad arguments (path doesn't exist, missing noxfile, …).                   |
| 3    | No programs observed in either run (test selection collected nothing).    |
| 4    | A `nox` invocation itself failed (see `run1/test.log` / `run2/test.log`). |

## Flags

```
--icon4py PATH                          icon4py checkout, abs or rel (required)
--session NAME                          nox session name (default: test_model)
--selection NAME                        noxfile selection (required); validated against
                                        icon4py's actual noxfile at runtime
--component NAME                        leaf subpackage name (required); validated
                                        against icon4py's actual noxfile at runtime
--python X.Y                            python version for the nox session (default: 3.10)
--workdir PATH                          where run1/, run2/, diffs/, report.txt land,
                                        abs or rel (default: <icon4py>/_dace_deterministic_codegen/).
                                        Wiped before each run.
--posarg ARG                            forwarded to pytest. Repeatable.
```

## CI integration

The harness runs in CSCS CI as a separate `dace-determinism` stage,
defined in `ci/cscs-ci-dace-determinism.yml` and wired into the
pipeline via `ci/cscs-ci.yml`. A small driver script,
`ci/dace_deterministic_codegen/run_in_ci.sh`, encapsulates the
clone + bootstrap + harness invocation so the YAML stays minimal and
the same flow can be reproduced locally.

### Reproducing a CI run locally

The driver script reads only env vars, so a green or red CI run can
be reproduced one-to-one by exporting the same variables and invoking
`run_in_ci.sh`:

```bash
# gt4py CI venv with editable gt4py already there
source /path/to/gt4py-venv/bin/activate

export GT4PY_PATH=/path/to/gt4py
export ICON4PY_REPO=https://github.com/C2SM/icon4py.git
export ICON4PY_REF=main          # or the SHA from the failing run
export ICON4PY_PATH=/tmp/icon4py

# Optional: custom dace branch
export DACE_REPO=https://github.com/GridTools/dace.git
export DACE_REF=dace_toolchain_deterministic
export DACE_PATH=/tmp/dace

export DACE_DETERMINISM_SELECTION=stencils
export DACE_DETERMINISM_COMPONENT=muphys
export DACE_DETERMINISM_PYTHON=3.10
export DACE_DETERMINISM_BACKEND=dace_cpu
export DACE_DETERMINISM_GRID=icon_regional

bash $GT4PY_PATH/ci/dace_deterministic_codegen/run_in_ci.sh
```
