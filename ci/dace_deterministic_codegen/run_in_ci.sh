#!/usr/bin/env bash
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# Driver for running the dace_deterministic_codegen harness in CI.
#
# Encapsulates the clone + bootstrap + harness invocation so the YAML
# stays minimal and the logic is easy to reproduce locally (just set
# the env vars and run the script).
#
# Required environment variables (CI sets all of these via job vars):
#   GT4PY_PATH                       Existing gt4py checkout (the commit under test).
#   ICON4PY_REPO                     Git URL to clone icon4py from.
#   ICON4PY_REF                      Git ref (branch, tag, or SHA) to checkout.
#   ICON4PY_PATH                     Where to clone icon4py to (created if missing).
#   DACE_DETERMINISM_SELECTION       icon4py noxfile selection: stencils|datatest|basic.
#   DACE_DETERMINISM_COMPONENT       icon4py subpackage leaf: muphys|dycore|...
#   DACE_DETERMINISM_PYTHON          Python version for the nox session: 3.10, 3.14, ...
#   DACE_DETERMINISM_BACKEND         dace_cpu | dace_gpu  (passed to pytest as --backend=...)
#   DACE_DETERMINISM_GRID            Grid name passed to pytest as --grid=...
#
# Optional environment variables:
#   DACE_REPO                        Git URL for a custom dace fork. When set,
#                                    DACE_REF and DACE_PATH must also be set.
#   DACE_REF                         Git ref of the custom dace branch.
#   DACE_PATH                        Where to clone dace to (created if missing).
#   DACE_DETERMINISM_WORKDIR         Where run1/, run2/, diffs/, report.txt land.
#                                    Default: ${ICON4PY_PATH}/_dace_deterministic_codegen
#   DACE_DETERMINISM_ARTIFACT_DIR    If set, the workdir is copied here at the end
#                                    (success or failure). Set to a path under
#                                    ${CI_PROJECT_DIR} for GitLab CI artifact upload.
#
# Custom dace branch behaviour:
#   - If DACE_REPO is unset, dace lands in the nox session venv via
#     icon4py's existing [tool.uv.sources] pin (currently the
#     GridTools/pypi published wheel). The parent venv may have its
#     own dace from the gt4py CI venv setup, but that's separate —
#     nox creates a fresh isolated venv and uv sync's into it from
#     icon4py's pyproject.toml.
#   - If DACE_REPO is set, the dace repo is cloned at DACE_REF and
#     icon4py's [tool.uv.sources] is patched to point at the clone.
#     Both the parent venv (Step 2) and the nox session venv (Step 3
#     onwards, via the patched source pin) end up with editable dace
#     from the same local path.
#
# Exit codes: passed through from dace_deterministic_codegen.py.
#   0 = deterministic, 1 = differs, 2/3/4 = harness errors.
#   See the harness README for the full table.

set -euo pipefail

# --- Validate required env vars -------------------------------------------
required=(
    GT4PY_PATH
    ICON4PY_REPO
    ICON4PY_REF
    ICON4PY_PATH
    DACE_DETERMINISM_SELECTION
    DACE_DETERMINISM_COMPONENT
    DACE_DETERMINISM_PYTHON
    DACE_DETERMINISM_BACKEND
    DACE_DETERMINISM_GRID
)
missing=()
for v in "${required[@]}"; do
    if [[ -z "${!v:-}" ]]; then
        missing+=("$v")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "error: missing required env vars: ${missing[*]}" >&2
    exit 2
fi

# Custom dace branch is all-or-nothing: setting one of the three
# DACE_* vars without the others would leave us in a half-configured
# state where it's unclear whether the local dace is supposed to win
# over icon4py's source pin.
if [[ -n "${DACE_REPO:-}" ]]; then
    if [[ -z "${DACE_REF:-}" || -z "${DACE_PATH:-}" ]]; then
        echo "error: DACE_REPO is set but DACE_REF and/or DACE_PATH are not." >&2
        echo "       To use a custom dace branch, set all three together:" >&2
        echo "         DACE_REPO   - git URL of the dace fork" >&2
        echo "         DACE_REF    - branch, tag, or SHA to check out" >&2
        echo "         DACE_PATH   - where to clone dace (typically \${WORKDIR}/dace)" >&2
        exit 2
    fi
fi

# Active venv check. The Docker image sets VIRTUAL_ENV; bare local runs
# might not. We don't auto-activate — that's the caller's responsibility —
# but warn if it's missing, since installing into the system Python is
# almost never what's wanted.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "warning: VIRTUAL_ENV is not set. Activate the gt4py CI venv first" >&2
    echo "         (the gt4py CI Docker image sets this automatically; only" >&2
    echo "         relevant when running this script outside the CI image)." >&2
fi

DACE_DETERMINISM_WORKDIR_DEFAULT="${ICON4PY_PATH}/_dace_deterministic_codegen"
DACE_DETERMINISM_WORKDIR="${DACE_DETERMINISM_WORKDIR:-${DACE_DETERMINISM_WORKDIR_DEFAULT}}"

HARNESS_DIR="${GT4PY_PATH}/ci/dace_deterministic_codegen"
HARNESS="${HARNESS_DIR}/dace_deterministic_codegen.py"
BOOTSTRAP="${HARNESS_DIR}/bootstrap_icon4py.py"

if [[ ! -f "$HARNESS" ]]; then
    echo "error: harness not found at $HARNESS" >&2
    echo "       (is GT4PY_PATH=$GT4PY_PATH the gt4py repo root?)" >&2
    exit 2
fi
if [[ ! -f "$BOOTSTRAP" ]]; then
    echo "error: bootstrap not found at $BOOTSTRAP" >&2
    exit 2
fi

# --- Helper: shallow-clone a repo at a ref, with SHA fallback ------------
# Some git versions can't combine --depth 1 with arbitrary commit SHAs in
# `clone -b`. If -b fails, fall back to a full clone + explicit checkout.
clone_at_ref() {
    local repo="$1" ref="$2" dest="$3" label="$4"
    if [[ -d "${dest}/.git" ]]; then
        echo "    (${label} already cloned at ${dest}; fetching ${ref})"
        git -C "${dest}" fetch --depth 1 origin "${ref}"
        git -C "${dest}" checkout FETCH_HEAD
        return
    fi
    if ! git clone --depth 1 -b "${ref}" "${repo}" "${dest}" 2>/dev/null; then
        echo "    (-b ${ref} failed; ${ref} may be a SHA — doing full clone + checkout)"
        git clone "${repo}" "${dest}"
        git -C "${dest}" checkout "${ref}"
    fi
}

# --- Step 1: clone icon4py at the pinned ref -----------------------------
echo "==> [1/4] cloning icon4py @ ${ICON4PY_REF} from ${ICON4PY_REPO}"
clone_at_ref "${ICON4PY_REPO}" "${ICON4PY_REF}" "${ICON4PY_PATH}" "icon4py"

# --- Step 1b (optional): clone custom dace -------------------------------
if [[ -n "${DACE_REPO:-}" ]]; then
    echo "==> [1b/4] cloning dace @ ${DACE_REF} from ${DACE_REPO}"
    clone_at_ref "${DACE_REPO}" "${DACE_REF}" "${DACE_PATH}" "dace"
fi

# --- Step 2: install editable gt4py (+ dace) + tomli_w into the venv -----
# The gt4py CI Docker image already has gt4py's deps installed via uv
# sync --no-install-project. We add gt4py itself (editable, pointing at
# our checkout), tomli_w (which bootstrap_icon4py.py imports), and
# optionally dace (editable, when a custom branch is being tested).
# --no-deps skips re-resolving heavy transitive deps; the icon4py
# bootstrap below will handle anything missing via uv sync --active.
if [[ -n "${DACE_PATH:-}" ]]; then
    echo "==> [2/4] installing editable gt4py + dace + tomli_w into ${VIRTUAL_ENV:-system Python}"
else
    echo "==> [2/4] installing editable gt4py + tomli_w into ${VIRTUAL_ENV:-system Python}"
fi
uv pip install --no-deps -e "${GT4PY_PATH}"
if [[ -n "${DACE_PATH:-}" ]]; then
    uv pip install --no-deps -e "${DACE_PATH}"
fi
uv pip install tomli_w

# --- Step 3: bootstrap icon4py into the active venv ----------------------
# Patches icon4py's [tool.uv.sources] so gt4py (and optionally dace)
# resolve to our local checkouts, then `uv lock` + `uv sync --active`.
# This is what makes the editable installs survive when icon4py's noxfile
# creates its session venv and runs `uv sync` inside it — that uv sync
# sees the patched source pins and installs editable from the same paths.
echo "==> [3/4] bootstrapping icon4py into the active venv"
bootstrap_args=( --icon4py "${ICON4PY_PATH}" --gt4py "${GT4PY_PATH}" )
if [[ -n "${DACE_PATH:-}" ]]; then
    bootstrap_args+=( --dace "${DACE_PATH}" )
fi
python "${BOOTSTRAP}" "${bootstrap_args[@]}"

# --- Step 4: run the determinism harness ---------------------------------
echo "==> [4/4] running the determinism harness"
echo "    selection=${DACE_DETERMINISM_SELECTION} component=${DACE_DETERMINISM_COMPONENT}"
echo "    python=${DACE_DETERMINISM_PYTHON} backend=${DACE_DETERMINISM_BACKEND} grid=${DACE_DETERMINISM_GRID}"
echo "    workdir=${DACE_DETERMINISM_WORKDIR}"

# Run with `set +e` and capture the exit code so the artifact-copy step
# below runs whether the harness reported determinism, non-determinism,
# or a tooling error. The harness is the source of truth on the exit
# code; we just defer reacting to it.
set +e
python "${HARNESS}" \
    --icon4py   "${ICON4PY_PATH}" \
    --selection "${DACE_DETERMINISM_SELECTION}" \
    --component "${DACE_DETERMINISM_COMPONENT}" \
    --python    "${DACE_DETERMINISM_PYTHON}" \
    --workdir   "${DACE_DETERMINISM_WORKDIR}" \
    --posarg=--backend="${DACE_DETERMINISM_BACKEND}" \
    --posarg=--grid="${DACE_DETERMINISM_GRID}"
harness_rc=$?
set -e

# --- Step 5 (optional): publish artifacts --------------------------------
# If DACE_DETERMINISM_ARTIFACT_DIR is set (typically in CI to a path
# under ${CI_PROJECT_DIR}), copy the workdir there so GitLab can pick
# it up as a build artifact. We do this whether the harness passed or
# failed — both outcomes have a useful report.txt.
if [[ -n "${DACE_DETERMINISM_ARTIFACT_DIR:-}" ]]; then
    echo "==> publishing artifacts to ${DACE_DETERMINISM_ARTIFACT_DIR}"
    rm -rf "${DACE_DETERMINISM_ARTIFACT_DIR}"
    mkdir -p "$(dirname "${DACE_DETERMINISM_ARTIFACT_DIR}")"
    if [[ -d "${DACE_DETERMINISM_WORKDIR}" ]]; then
        cp -r "${DACE_DETERMINISM_WORKDIR}" "${DACE_DETERMINISM_ARTIFACT_DIR}"
    else
        # Harness errored before creating the workdir — leave a note so
        # the artifact upload still has something for diagnosis from the
        # GitLab UI without ssh'ing to the runner.
        mkdir -p "${DACE_DETERMINISM_ARTIFACT_DIR}"
        cat > "${DACE_DETERMINISM_ARTIFACT_DIR}/MISSING_WORKDIR.txt" <<NOTE
The dace_deterministic_codegen harness exited with code ${harness_rc}
before creating its workdir at:
  ${DACE_DETERMINISM_WORKDIR}

Likely causes: invalid --selection/--component (rc 2), missing icon4py
noxfile (rc 2), or a failure during bootstrap. See the job log for the
actual error message.
NOTE
    fi
fi

exit $harness_rc
