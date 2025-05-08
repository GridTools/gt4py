#! /usr/bin/env -S uv run -q --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
# /// script
# dependencies = ["nox>=2025.02.09"]
# ///

from __future__ import annotations

import pathlib
import sys
import types
from collections.abc import Sequence
from unittest import mock
from typing import Any, Final, Literal, TypeAlias

import nox

# When running the `noxfile.py` as a script, the current directory is
# added to the `sys.path`, so we can import other python file in this
# directory directly, like `noxfile_utils.py`.
# However, when running `nox` as a CLI tool, the current
# directory is not added to the `sys.path`, so we need to add it manually.
if (_folder := f"{pathlib.Path(__file__).parent!s}") not in sys.path:
    sys.path.insert(0, _folder)

import noxfile_utils as nox_utils


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
    "test_eve-3.10",
    "test_eve-3.11",
    "test_next-3.10(internal, cpu, nomesh)",
    "test_next-3.10(dace, cpu, nomesh)",
    "test_next-3.11(internal, cpu, nomesh)",
    "test_next-3.11(dace, cpu, nomesh)",
    "test_package-3.10",
    "test_package-3.11",
    "test_storage-3.10(cpu)",
    "test_storage-3.11(cpu)",
]


PYTHON_VERSIONS: Final[list[str]] = pathlib.Path(".python-versions").read_text().splitlines()

# -- Parameter sets --
DeviceOption: TypeAlias = Literal["cpu", "cuda11", "cuda12", "rocm4_3", "rocm5_0"]
DeviceNoxParam: Final = types.SimpleNamespace(
    **{device: nox.param(device, id=device, tags=[device]) for device in DeviceOption.__args__}
)
DeviceTestSettings: Final[dict[str, dict[str, Sequence]]] = {
    "cpu": {"extras": [], "markers": ["not requires_gpu"]},
    **{
        device: {"extras": [device], "markers": ["requires_gpu"]}
        for device in ["cuda11", "cuda12", "rocm4_3", "rocm5_0"]
    },
}

CodeGenOption: TypeAlias = Literal["internal", "dace"]
CodeGenNoxParam: Final = types.SimpleNamespace(
    **{
        codegen: nox.param(codegen, id=codegen, tags=[codegen])
        for codegen in CodeGenOption.__args__
    }
)
CodeGenTestSettings: Final[dict[str, dict[str, Sequence]]] = {
    "internal": {"extras": [], "markers": ["not requires_dace"]},
    "dace": {"extras": ["dace"], "markers": ["requires_dace"]},
}
# Use dace-next for GT4Py-next, to install a different dace version than in cartesian
CodeGenNextTestSettings = CodeGenTestSettings | {
    "dace": {"extras": ["dace-next"], "markers": ["requires_dace"]},
}


# -- Sessions --
@nox_utils.customize_session(
    python=PYTHON_VERSIONS,
    tags=["cartesian"],
    ignore_paths=[  # In CI mode, skip when only gt4py.next or doc files have been updated
        "src/gt4py/next/*",
        "tests/next_tests/**",
        "examples/**",
        "*.md",
        "*.rst",
    ],
)
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
@nox.parametrize("codegen", [CodeGenNoxParam.internal, CodeGenNoxParam.dace])
def test_cartesian(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.cartesian' tests."""

    codegen_settings = CodeGenTestSettings[codegen]
    device_settings = DeviceTestSettings[device]

    nox_utils.install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=["test"],
    )

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


@nox_utils.customize_session(
    python=PYTHON_VERSIONS,
    tags=["cartesian", "next", "cpu"],
    paths=[  # In CI mode, run when gt4py.eve files (or package settings) are changed
        "src/gt4py/eve/*",
        "tests/eve_tests/*",
        ".github/workflows/*",
        "*.lock",
        "*.toml",
        "*.yml",
        "noxfile*.py",
    ],
)
def test_eve(session: nox.Session) -> None:
    """Run 'gt4py.eve' tests."""

    nox_utils.install_session_venv(session, groups=["test"])

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

    nox_utils.install_session_venv(session, extras=["testing"], groups=["docs", "test"])

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


@nox_utils.customize_session(
    python=PYTHON_VERSIONS,
    tags=["next"],
    ignore_paths=[  # In CI mode, skip when only gt4py.cartesian or doc files have been updated
        "src/gt4py/cartesian/**",
        "tests/cartesian_tests/**",
        "examples/**",
        "*.md",
        "*.rst",
    ],
)
@nox.parametrize(
    "meshlib",
    [
        nox.param("nomesh", id="nomesh", tags=["nomesh"]),
        nox.param("atlas", id="atlas", tags=["atlas"]),
    ],
)
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
@nox.parametrize("codegen", [CodeGenNoxParam.internal, CodeGenNoxParam.dace])
def test_next(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
    meshlib: Literal["nomesh", "atlas"],
) -> None:
    """Run selected 'gt4py.next' tests."""

    codegen_settings = CodeGenNextTestSettings[codegen]
    device_settings = DeviceTestSettings[device]
    groups: list[str] = ["test"]
    mesh_markers: list[str] = []

    match meshlib:
        case "nomesh":
            mesh_markers.append("not requires_atlas")
        case "atlas":
            mesh_markers.append("requires_atlas")
            groups.append("frameworks")

    nox_utils.install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=groups,
    )

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

    nox_utils.install_session_venv(session, groups=["test"])

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


@nox_utils.customize_session(
    python=PYTHON_VERSIONS,
    tags=["cartesian", "next"],
    paths=[  # In CI mode, run when gt4py.storage files (or package settings) are changed
        "src/gt4py/storage/**",
        "src/gt4py/cartesian/backend/**",  # For DaCe storages
        "tests/storage_tests/**",
        ".github/workflows/**",
        "*.lock",
        "*.toml",
        "*.yml",
        "noxfile*.py",
    ],
)
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
def test_storage(
    session: nox.Session,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.storage' tests."""

    device_settings = DeviceTestSettings[device]

    nox_utils.install_session_venv(
        session, extras=["performance", "testing", *device_settings["extras"]], groups=["test"]
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


@nox.session(python=False)
def _is_required_by_repo_changes(session: nox.Session) -> None:
    """
    Evaluate if given sessions (session names passed as 'posargs'`) are required by changes in the repo.

    Use `CI_NOX_RUN_ONLY_IF_CHANGED_FROM` env variable to pass the reference commit.

    Example:
        $ CI_NOX_RUN_ONLY_IF_CHANGED_FROM='main' CI_NOX_VERBOSE=1 nox -s _is_required_by_repo_changes -- test_cartesian
    """

    for arg in session.posargs:
        if arg.startswith("--"):
            session.error(f"Invalid argument: {arg}")
        print(f"{arg}: {nox_utils.is_required_by_repo_changes(arg, verbose=True)}")


if __name__ == "__main__":
    nox.main()
