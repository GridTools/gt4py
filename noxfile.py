# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
import types
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import nox

try:
    import pytest

    NO_TESTS_COLLECTED_EXIT_CODE = pytest.ExitCode.NO_TESTS_COLLECTED
except ImportError:
    NO_TESTS_COLLECTED_EXIT_CODE = 5

# -- nox configuration --
nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "cartesian_tests-3.10(internal, cpu)",
    "cartesian_tests-3.10(dace, cpu)",
    "cartesian_tests-3.11(internal, cpu)",
    "cartesian_tests-3.11(dace, cpu)",
    "eve_tests-3.10",
    "eve_tests-3.11",
    "next_tests-3.10(internal, cpu, nomesh)",
    "next_tests-3.10(dace, cpu, nomesh)",
    "next_tests-3.11(internal, cpu, nomesh)",
    "next_tests-3.11(dace, cpu, nomesh)",
    "storage_tests-3.10(cpu)",
    "storage_tests-3.11(cpu)",
]

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


# -- nox sessions --
@nox.session(python=["3.10", "3.11"], tags=["cartesian"])
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
@nox.parametrize("codegen", [CodeGenNoxParam.internal, CodeGenNoxParam.dace])
def cartesian_tests(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.cartesian' tests."""

    codegen_settings = CodeGenTestSettings[codegen]
    device_settings = DeviceTestSettings[device]

    _install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=["test"],
    )

    num_processes = session.env.get("NUM_PROCESSES", "auto")
    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"])

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "cartesian_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules --doctest-ignore-import-errors -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "cartesian"),
    )


@nox.session(python=["3.10", "3.11"], tags=["cartesian", "next", "cpu"])
def eve_tests(session: nox.Session) -> None:
    """Run 'gt4py.eve' tests."""

    _install_session_venv(session, groups=["test"])

    num_processes = session.env.get("NUM_PROCESSES", "auto")

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
        str(pathlib.Path("tests") / "eve_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "eve"),
    )


@nox.session(python=["3.10", "3.11"], tags=["next"])
@nox.parametrize(
    "meshlib",
    [
        nox.param("nomesh", id="nomesh", tags=["nomesh"]),
        nox.param("atlas", id="atlas", tags=["atlas"]),
    ],
)
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
@nox.parametrize("codegen", [CodeGenNoxParam.internal, CodeGenNoxParam.dace])
def next_tests(
    session: nox.Session,
    codegen: CodeGenOption,
    device: DeviceOption,
    meshlib: Literal["nomesh", "atlas"],
) -> None:
    """Run selected 'gt4py.next' tests."""

    codegen_settings = CodeGenTestSettings[codegen]
    device_settings = DeviceTestSettings[device]
    mesh_markers = []
    match meshlib:
        case "nomesh":
            mesh_markers.append("not requires_atlas")
        case "atlas":
            mesh_markers.append("requires_atlas")

    _install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=["test"],
    )

    num_processes = session.env.get("NUM_PROCESSES", "auto")
    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"] + mesh_markers)

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "next_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "next"),
    )


@nox.session(python=["3.10", "3.11"], tags=["cartesian", "next"])
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
def storage_tests(
    session: nox.Session,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.storage' tests."""

    device_settings = DeviceTestSettings[device]

    _install_session_venv(
        session, extras=["performance", "testing", *device_settings["extras"]], groups=["test"]
    )

    num_processes = session.env.get("NUM_PROCESSES", "auto")
    markers = " and ".join(device_settings["markers"])

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "storage_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "storage"),
        success_codes=[NO_TESTS_COLLECTED_EXIT_CODE],
    )


# -- utils --
def _install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""
    uv_venv = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("NOX")
        or key.startswith("PYTEST_")
        or key.startswith("PYTHON")
        or key.startswith("UV_")
    } | {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    session.run_install(
        "uv",
        "sync",
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env=uv_venv,
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        )
