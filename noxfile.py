#!/usr/bin/env -S uv run -q --script --python 3.12
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# Note:
#   The explicit '--python 3.11' in the shebang is only needed due
#   to the existence of the .python-versions file, which overrides
#   the PEP 723 'requires-python' metadata.
# /// script
# requires-python = ">=3.11"
# dependencies = ["nox>=2025.02.09", "uv>=0.6.10", "tomli; python_version < '3.11'"]
# ///

from __future__ import annotations

import os
import pathlib
import shutil
import sys
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import nox


try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


# This is needed because uv now fails to create an env when it already exists.
# See: https://github.com/astral-sh/uv/issues/17899
os.environ["UV_VENV_CLEAR"] = "1"

# This should just be `pytest.ExitCode.NO_TESTS_COLLECTED` but `pytest`
# is not guaranteed to be available in the venv where `nox` is running.
NO_TESTS_COLLECTED_EXIT_CODE: Final = 5

# -- nox configuration --
nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "test_cartesian-3.10(internal, cpu)",
    "test_cartesian-3.10(dace, cpu)",
    "test_cartesian-3.11(internal, cpu)",
    "test_cartesian-3.11(dace, cpu)",
    "test_cartesian-3.12(internal, cpu)",
    "test_cartesian-3.12(dace, cpu)",
    "test_cartesian-3.13(internal, cpu)",
    "test_cartesian-3.13(dace, cpu)",
    "test_cartesian-3.14(internal, cpu)",
    "test_cartesian-3.14(dace, cpu)",
    "test_eve-3.10",
    "test_eve-3.11",
    "test_eve-3.12",
    "test_eve-3.13",
    "test_eve-3.14",
    "test_next-3.10(internal, cpu, nomesh)",
    "test_next-3.10(dace, cpu, nomesh)",
    "test_next-3.11(internal, cpu, nomesh)",
    "test_next-3.11(dace, cpu, nomesh)",
    "test_next-3.12(internal, cpu, nomesh)",
    "test_next-3.12(dace, cpu, nomesh)",
    "test_next-3.13(internal, cpu, nomesh)",
    "test_next-3.13(dace, cpu, nomesh)",
    "test_next-3.14(internal, cpu, nomesh)",
    "test_next-3.14(dace, cpu, nomesh)",
    "test_package-3.10",
    "test_package-3.11",
    "test_package-3.12",
    "test_package-3.13",
    "test_package-3.14",
    "test_storage-3.10(cpu)",
    "test_storage-3.11(cpu)",
    "test_storage-3.12(cpu)",
    "test_storage-3.13(cpu)",
    "test_storage-3.14(cpu)",
]

REPO_ROOT: Final = pathlib.Path(__file__).parent.resolve().absolute()
PYTHON_VERSIONS: Final[list[str]] = [
    v
    for line in (REPO_ROOT / ".python-versions").read_text().splitlines()
    if (v := line.strip()) and not v.startswith("#")
]
REQUIRES_PYTHON = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
    "requires-python"
]

# -- Parameter sets --
DeviceOption: TypeAlias = Literal["cpu", "cuda12", "cuda13", "rocm6", "rocm7"]
DeviceNoxParam: Final[dict[DeviceOption, nox.param]] = {
    device: nox.param(device, id=device, tags=[device]) for device in DeviceOption.__args__
}
DeviceTestSettings: Final[dict[DeviceOption, dict[str, list[str]]]] = {
    "cpu": {"extras": [], "markers": ["not requires_gpu"]},
    **{
        device: {"extras": [device], "markers": ["requires_gpu"]}
        for device in DeviceOption.__args__
        if device != "cpu"
    },
}

CodeGenOption: TypeAlias = Literal["internal", "dace"]
CodeGenNoxParam: Final[dict[CodeGenOption, nox.param]] = {
    codegen: nox.param(codegen, id=codegen, tags=[codegen]) for codegen in CodeGenOption.__args__
}
CodeGenTestSettings: Final[dict[str, dict[str, list[str]]]] = {
    "internal": {"extras": ["jax"], "markers": ["not requires_dace"]}
}
CodeGenDaceTestSettings = CodeGenTestSettings | {
    "dace": {"extras": [], "markers": ["requires_dace"]},
}


# -- Utilities --
def install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """
    Install session packages using the `uv` tool.

    Args:
        session: The Nox session object.
        *args: Additional packages to install in the session (via `uv pip install`)
        extras: Names of package's extras to install.
        groups: Names of dependency groups to install.
    """
    session.run_install(
        "uv",
        "sync",
        "--python",
        # uv does not yet combine explicit python version requests with the
        # `requires-python` range in `pyproject.toml`, so we do it manually.
        # See: https://github.com/astral-sh/uv/issues/16654
        f"{REQUIRES_PYTHON}, ~={session.python!s}.0",
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env=session.env | dict(UV_PROJECT_ENVIRONMENT=session.virtualenv.location),
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env=session.env | dict(UV_PROJECT_ENVIRONMENT=session.virtualenv.location),
        )


# -- Sessions --
@nox.session(python=PYTHON_VERSIONS, tags=["cartesian"])
@nox.parametrize("device", [*DeviceNoxParam.values()])
@nox.parametrize("codegen", [*CodeGenNoxParam.values()])
def test_cartesian(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.cartesian' tests."""

    codegen_settings = CodeGenDaceTestSettings[codegen]
    device_settings = DeviceTestSettings[device]
    extras = [
        "standard",
        "testing",
        *codegen_settings.get("extras", []),
        *device_settings.get("extras", []),
    ]
    groups = ["test", *codegen_settings.get("groups", []), *device_settings.get("groups", [])]

    install_session_venv(session, extras=extras, groups=groups)

    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"])

    session.run(
        *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "cartesian_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules --doctest-ignore-import-errors -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "cartesian"),
    )


@nox.session(python=PYTHON_VERSIONS, tags=["cartesian", "next", "cpu"])
def test_eve(session: nox.Session) -> None:
    """Run 'gt4py.eve' tests."""

    install_session_venv(session, groups=["test"])

    session.run(
        *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
        str(pathlib.Path("tests") / "eve_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "eve"),
    )


@nox.session(python=PYTHON_VERSIONS, tags=["next"])
def test_examples(session: nox.Session) -> None:
    """Run and test documentation workflows."""

    install_session_venv(session, extras=["testing"], groups=["docs", "test"])

    session.run(*"jupytext docs/user/next/QuickstartGuide.md --to .ipynb".split())
    session.run(*"jupytext docs/user/next/advanced/*.md --to .ipynb".split())

    for notebook, extra_args in [
        ("docs/user/next/workshop/slides", None),
        ("docs/user/next/workshop/exercises", ["-k", "solutions"]),
        ("docs/user/next/QuickstartGuide.ipynb", None),
        ("docs/user/next/advanced", None),
        ("examples", (None)),
    ]:
        session.run(
            *f"pytest --nbmake {notebook} -sv -n 1 --benchmark-disable".split(),
            *(extra_args or []),
        )


@nox.session(python=PYTHON_VERSIONS, tags=["next"])
@nox.parametrize(
    "meshlib",
    [
        nox.param("nomesh", id="nomesh", tags=["nomesh"]),
        nox.param("atlas", id="atlas", tags=["atlas"]),
    ],
)
@nox.parametrize("device", [*DeviceNoxParam.values()])
@nox.parametrize("codegen", [*CodeGenNoxParam.values()])
def test_next(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
    meshlib: Literal["nomesh", "atlas"],
) -> None:
    """Run selected 'gt4py.next' tests."""

    codegen_settings = CodeGenDaceTestSettings[codegen]
    device_settings = DeviceTestSettings[device]
    extras = [
        "standard",
        "testing",
        *codegen_settings.get("extras", []),
        *device_settings.get("extras", []),
    ]
    groups = ["test", *codegen_settings.get("groups", []), *device_settings.get("groups", [])]
    mesh_markers: list[str] = []

    match meshlib:
        case "nomesh":
            mesh_markers.append("not requires_atlas")
        case "atlas":
            mesh_markers.append("requires_atlas")
            groups.append("frameworks")

    install_session_venv(session, extras=extras, groups=groups)

    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"] + mesh_markers)

    session.run(
        *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "next_tests"),
        *session.posargs,
        success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
    )
    session.run(
        *"pytest --doctest-modules --doctest-ignore-import-errors -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "next"),
        success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
    )


@nox.session(python=PYTHON_VERSIONS, tags=["cartesian", "next", "cpu"])
def test_package(session: nox.Session) -> None:
    """Run 'gt4py' package level tests."""

    install_session_venv(session, groups=["test"])

    session.run(
        *"pytest --cache-clear -sv".split(),
        str(pathlib.Path("tests") / "package_tests"),
        *session.posargs,
    )

    modules = [str(path) for path in (pathlib.Path("src") / "gt4py").glob("*.py")]
    session.run(
        *"pytest --doctest-modules --doctest-ignore-import-errors -sv".split(),
        *modules,
        success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
    )


@nox.session(python=PYTHON_VERSIONS, tags=["cartesian", "next"])
@nox.parametrize("device", [*DeviceNoxParam.values()])
def test_storage(
    session: nox.Session,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.storage' tests."""

    device_settings = DeviceTestSettings[device]

    install_session_venv(
        session, extras=["standard", "testing", *device_settings["extras"]], groups=["test"]
    )

    markers = " and ".join(device_settings["markers"])

    session.run(
        *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "storage_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "storage"),
        success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
    )


@nox.session(python=PYTHON_VERSIONS, tags=["next"])
def test_typing_exports(session: nox.Session) -> None:
    """Test GT4Py usability in a typed client context."""
    install_session_venv(session, extras=["standard"], groups=["test", "typing_exports"])

    session.run(
        "pytest",
        "-sv",
        "--mypy-testing-base",
        "typing_tests",
        "typing_tests",
        *session.posargs,
    )


# -- DaCe codegen determinism check --
#
# The `test_next_dace_determinism` session below runs gt4py's pytest
# selection twice with an isolated GT4PY_BUILD_CACHE_DIR per run, then
# verifies the DaCe-generated source files under <program>/src/ are
# byte-identical between the two runs. A diff is a determinism bug
# somewhere in the gt4py + dace toolchain for that test selection.
#
# Comparison logic (snapshot, hash, diff, report) lives in
# `scripts/dace_deterministic_codegen.py`; the helper below just
# wires gt4py's existing pytest invocation pattern into a "run
# twice + compare" loop.
#
# Workdir at REPO_ROOT/_dace_deterministic_codegen/ (wiped before
# each session invocation):
#   run1/.gt4py_cache/...    (first run's cached programs)
#   run2/.gt4py_cache/...    (second run's cached programs)
#   diffs/<program>/<file>.diff   (only on mismatch)
#   report.txt                    (human-readable summary)
#
# Only `dace` codegen is checked (`internal` doesn't go through dace),
# so the codegen parameter is dropped from this session's signature.

DACE_DETERMINISM_WORKDIR_NAME: Final = "_dace_deterministic_codegen"


def _run_dace_determinism_check(
    session: nox.Session,
    pytest_args: Sequence[str],
) -> None:
    """Run pytest twice with an isolated cache per run, then verify the
    DaCe-generated source files are byte-identical between the two runs.

    Each run sets ``GT4PY_BUILD_CACHE_DIR=<run_dir>`` so the gt4py.next cache
    lands at ``<run_dir>/.gt4py_cache/``, where the comparator walks each
    program folder's ``src/{cpu,cuda}/...`` sub-tree.

    On mismatch, calls ``session.error(...)`` with a pointer to the diffs/
    directory and report.txt so the failure is actionable.
    """
    workdir = REPO_ROOT / DACE_DETERMINISM_WORKDIR_NAME
    if workdir.exists():
        shutil.rmtree(workdir)

    run1_dir = workdir / "run1"
    run2_dir = workdir / "run2"
    run1_dir.mkdir(parents=True)
    run2_dir.mkdir(parents=True)

    cache_subdir = ".gt4py_cache"

    dacecache = REPO_ROOT / ".dacecache"
    dacecache_preexisted = dacecache.exists()

    def env_for_run(run_dir: pathlib.Path) -> dict[str, str]:
        # gt4py.next appends `.gt4py_cache` to GT4PY_BUILD_CACHE_DIR, so we
        # pass the parent directory and the cache lands at .gt4py_cache/
        # underneath. Setting GT4PY_BUILD_CACHE_LIFETIME to `persistent` keeps
        # the cache around long enough for the snapshot pass to read it.
        #
        # Setting DACE_compiler_build_folder_mode to `development` is REQUIRED:
        # gt4py configures dace to `production` mode by default, which cleans
        # up the dace build folder after compilation — leaving only the
        # compiled .so and stripping the codegen sources we need to diff.
        # Forcing `development` keeps `src/...` around so the checker has
        # codegen to compare. (See src/gt4py/next/program_processors/runners/
        # dace/workflow/common.py for the upstream config this overrides; the
        # comment there documents this env var as the escape hatch.)
        return {
            "GT4PY_BUILD_CACHE_DIR": str(run_dir),
            "GT4PY_BUILD_CACHE_LIFETIME": "persistent",
            "DACE_compiler_build_folder_mode": "development",
        }

    for run_dir in (run1_dir, run2_dir):
        session.run(
            *pytest_args,
            *session.posargs,
            env=session.env | env_for_run(run_dir),
            # The determinism check cares only about whether the DaCe codegen
            # lands deterministically in the cache; individual test outcomes
            # are irrelevant. Failed tests (exit code 1) often reflect runtime
            # issues that have nothing to do with codegen.
            success_codes=[0, 1, NO_TESTS_COLLECTED_EXIT_CODE],
        )

    # Import the comparison library from scripts/. It uses only stdlib, so it
    # runs fine in nox's runtime python (no session venv needed).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.dace_deterministic_codegen import (
        DeterminismError,
        NoProgramsObservedError,
        NoSourceFilesObservedError,
        UnsupportedBackendError,
        check_determinism,
    )

    try:
        check_determinism(
            run1_dir / cache_subdir,
            run2_dir / cache_subdir,
            diffs_dir=workdir / "diffs",
            report_path=workdir / "report.txt",
            # Programs cached in only one run are reported but not counted as
            # determinism failures — see the success_codes note above for why
            # this is the right policy here.
            tolerate_missing=True,
        )
    except DeterminismError as e:
        session.error(f"{e}\nSee {workdir / 'report.txt'} and {workdir / 'diffs'}/")
    except NoProgramsObservedError as e:
        session.error(f"{e}\nLikely the pytest selection collected no tests.")
    except NoSourceFilesObservedError as e:
        session.error(str(e))
    except UnsupportedBackendError as e:
        session.error(str(e))
    finally:
        # Reclaim disk after the comparison. The two per-run caches are
        # ~hundreds of MB each in development mode. We always keep
        # `workdir/diffs/` and `workdir/report.txt` — those are the artifacts a
        # maintainer actually needs to debug a determinism failure; the raw
        # caches are reproducible by rerunning the session.
        cleanup_targets = [run1_dir, run2_dir]
        if not dacecache_preexisted:
            cleanup_targets.append(dacecache)
        for tbd in cleanup_targets:
            if tbd.exists():
                session.log(f"cleanup: removing {tbd}")
                shutil.rmtree(tbd, ignore_errors=True)


@nox.session(python=PYTHON_VERSIONS, tags=["next", "dace", "determinism"])
@nox.parametrize(
    "meshlib",
    [
        nox.param("nomesh", id="nomesh", tags=["nomesh"]),
        nox.param("atlas", id="atlas", tags=["atlas"]),
    ],
)
@nox.parametrize("device", [*DeviceNoxParam.values()])
def test_next_dace_determinism(
    session: nox.Session,
    device: DeviceOption,
    meshlib: Literal["nomesh", "atlas"],
) -> None:
    """Run selected 'gt4py.next' DaCe tests twice and verify codegen
    is byte-identical between the two runs."""

    codegen_settings = CodeGenDaceTestSettings["dace"]
    device_settings = DeviceTestSettings[device]
    extras = [
        "standard",
        "testing",
        *codegen_settings.get("extras", []),
        *device_settings.get("extras", []),
    ]
    groups = ["test", *codegen_settings.get("groups", []), *device_settings.get("groups", [])]
    mesh_markers: list[str] = []

    match meshlib:
        case "nomesh":
            mesh_markers.append("not requires_atlas")
        case "atlas":
            mesh_markers.append("requires_atlas")
            groups.append("frameworks")

    install_session_venv(session, extras=extras, groups=groups)

    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"] + mesh_markers)

    _run_dace_determinism_check(
        session,
        pytest_args=[
            *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
            "-m",
            f"{markers}",
            str(pathlib.Path("tests") / "next_tests"),
        ],
    )


if __name__ == "__main__":
    nox.main()
