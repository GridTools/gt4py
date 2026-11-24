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
#   The explicit '--python 3.12' in the shebang is only needed due
#   to the existence of the .python-versions file, which overrides
#   the PEP 723 'requires-python' metadata.
# /// script
# requires-python = ">=3.12"
# dependencies = ["nox>=2025.02.09", "uv>=0.6.10"]
# ///

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tomllib
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import nox


# This is needed because uv now fails to create an env when it already exists.
# See: https://github.com/astral-sh/uv/issues/17899
os.environ["UV_VENV_CLEAR"] = "1"

# This should just be `pytest.ExitCode.NO_TESTS_COLLECTED` but `pytest`
# is not guaranteed to be available in the venv where `nox` is running.
NO_TESTS_COLLECTED_EXIT_CODE: Final = 5

# -- nox configuration --
nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "test_cartesian-3.12(internal, cpu)",
    "test_cartesian-3.12(dace, cpu)",
    "test_cartesian-3.13(internal, cpu)",
    "test_cartesian-3.13(dace, cpu)",
    "test_cartesian-3.14(internal, cpu)",
    "test_cartesian-3.14(dace, cpu)",
    "test_eve-3.12",
    "test_eve-3.13",
    "test_eve-3.14",
    "test_next-3.12(internal, cpu, nomesh)",
    "test_next-3.12(dace, cpu, nomesh)",
    "test_next-3.13(internal, cpu, nomesh)",
    "test_next-3.13(dace, cpu, nomesh)",
    "test_next-3.14(internal, cpu, nomesh)",
    "test_next-3.14(dace, cpu, nomesh)",
    "test_package-3.12",
    "test_package-3.13",
    "test_package-3.14",
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
    "internal": {"extras": ["jax"], "markers": ["not uses_dace"]}
}
CodeGenDaceTestSettings = CodeGenTestSettings | {
    "dace": {"extras": [], "markers": ["uses_dace"]},
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


def _session_runs_torch(session: nox.Session, device: DeviceOption) -> bool:
    if device in ("rocm6", "rocm7"):
        # the torch wheels on PyPI are CUDA-only
        return False
    if sys.platform == "darwin":
        # the torch wheels carry their own `libomp.dylib`, which clashes with the Homebrew
        # one the compiled gtfn and dace extensions link: two OpenMP runtimes abort with
        # `OMP: Error #15`
        return False
    return True


# -- Sessions --
# Latest supported Python version, since it is the one with the newest Sphinx
@nox.session(python=PYTHON_VERSIONS[-1], tags=["docs"])
def docs(session: nox.Session) -> None:
    """Build the user documentation site (welcome page and `gt4py.cartesian` and `gt4py.next` subsites).

    The site is made of three independent Sphinx projects which are built into
    a single tree: the welcome page at the root, with the subsites in the
    `cartesian/` and `next/` subfolders. Positional arguments are forwarded to
    all `sphinx-build` invocations (e.g. `nox -s docs -- -W --keep-going`).
    """

    install_session_venv(session, extras=["standard", "testing"], groups=["docs"])

    docs_dir = REPO_ROOT / "docs" / "user"
    build_dir = docs_dir / "_build"
    html_dir = build_dir / "html"
    if html_dir.exists():
        shutil.rmtree(html_dir)

    # API reference of the `gt4py.cartesian` subsite: all subpackages except `next` and `eve`
    # (the `gt4py.next` subsite has no API reference)
    src_dir = REPO_ROOT / "src" / "gt4py"
    apidoc_dir = docs_dir / "cartesian" / "_source"
    for generated in apidoc_dir.glob("*.rst"):
        generated.unlink()
    session.run(
        *"sphinx-apidoc --private -q -o".split(),
        str(apidoc_dir),
        str(src_dir),
        str(src_dir / "next"),
        str(src_dir / "eve"),
    )

    for source, target in [(".", "."), ("cartesian", "cartesian"), ("next", "next")]:
        session.run(
            *"sphinx-build -b html -E -q".split(),
            *session.posargs,
            "-d",
            str(build_dir / "doctrees" / (target if target != "." else "root")),
            str(docs_dir / source),
            str(html_dir / target),
        )

    # Notebook execution artifacts are written by `myst-nb` next to the output folder
    shutil.rmtree(html_dir / "jupyter_execute", ignore_errors=True)
    (html_dir / ".nojekyll").touch()

    # Keep the old `latest/` URL working by redirecting it to the current site
    (html_dir / "latest").mkdir()
    (html_dir / "latest" / "index.html").write_text(
        '<!DOCTYPE html>\n<html>\n  <head>\n    <meta charset="utf-8">\n'
        '    <meta http-equiv="refresh" content="0; url=../index.html">\n'
        '    <link rel="canonical" href="../index.html">\n  </head>\n'
        '  <body><a href="../index.html">GT4Py documentation</a></body>\n</html>\n'
    )

    session.log(f"Documentation site built in '{html_dir}'")


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
        # An unmet `requires_*` marker means a broken environment. Fail loudly
        # instead of skipping all tests due to the auto-skip in 'tests/conftest.py'.
        # Applies to every `requires_*`: install the extra, or exclude the marker.
        "--require-optional-deps",
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

    # Pages of the `gt4py.next` documentation site which are executable notebooks
    next_docs_dir = pathlib.Path("docs") / "user" / "next"
    next_docs_pages = sorted(
        str(page)
        for page in next_docs_dir.rglob("*.md")
        if "workshop" not in page.parts and "format_name: myst" in page.read_text()
    )
    session.run("jupytext", *next_docs_pages, "--to", ".ipynb")
    session.run(*"jupytext docs/development/next/toolchain/*.md --to .ipynb".split())

    for notebook, extra_args in [
        ("docs/user/next/workshop/slides", None),
        ("docs/user/next/workshop/exercises", ["-k", "solution"]),
        (" ".join(page.removesuffix(".md") + ".ipynb" for page in next_docs_pages), None),
        ("docs/development/next/toolchain", None),
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

    torch_markers: list[str] = []
    if codegen == "internal" and _session_runs_torch(session, device):
        extras.append("torch")
    else:
        torch_markers.append("not requires_torch")

    install_session_venv(session, extras=extras, groups=groups)

    markers = " and ".join(
        codegen_settings["markers"] + device_settings["markers"] + mesh_markers + torch_markers
    )

    session.run(
        *"pytest --cache-clear -sv -n auto --dist loadgroup".split(),
        # An unmet `requires_*` marker means a broken environment. Fail loudly
        # instead of skipping all tests due to the auto-skip in 'tests/conftest.py'.
        # Applies to every `requires_*`: install the extra, or exclude the marker.
        "--require-optional-deps",
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "next_tests"),
        *session.posargs,
        # No `NO_TESTS_COLLECTED_EXIT_CODE` here: every combination selects tests,
        # so collecting none means the selection broke.
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
        # An unmet `requires_*` marker means a broken environment. Fail loudly
        # instead of skipping all tests due to the auto-skip in 'tests/conftest.py'.
        # Applies to every `requires_*`: install the extra, or exclude the marker.
        "--require-optional-deps",
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

    # Pass the config explicitly: with no '--config-file', mypy discovers one by
    # walking up from the plugin's temporary execution directory and reaches the
    # project's own '[tool.mypy]' table, which pins 'python_version' to the supported
    # floor and would collapse this session's 3.13/3.14 runs into the 3.12 one. See
    # the comments in 'typing_tests/mypy.ini'.
    session.run(
        "pytest",
        "-sv",
        "--mypy-testing-base",
        "typing_tests",
        "--mypy-ini-file",
        "typing_tests/mypy.ini",
        "typing_tests",
        *session.posargs,
    )


# -- DaCe codegen determinism check --
#
# The `test_next_dace_determinism` session below runs gt4py's pytest selection
# twice with an isolated GT4PY_BUILD_CACHE_DIR per run, then verifies the
# DaCe-generated source files are byte-identical between the two runs. A diff is
# a determinism bug somewhere in the gt4py + dace toolchain for that selection.
#
# The orchestration (run twice + compare) and the hashing/comparison library
# both live in `scripts/python/dace_determinism.py`, exposed as its
# `ci-check` and `check` subcommands. This session installs the `scripts`
# dependency group into the session venv and executes that script with the
# venv's own python (a gt4py-capable interpreter) — it does not import the
# module.
#
# Only `dace` codegen is checked (`internal` doesn't go through dace), so the
# codegen parameter is dropped from this session's signature.


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

    # The `scripts` group provides `typer`, needed by the dace-determinism CLI.
    install_session_venv(session, extras=extras, groups=[*groups, "scripts"])

    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"] + mesh_markers)

    # Execute the determinism script with the session venv's own python (resolved
    # by nox from "python"); the script runs the pytest selection twice via
    # `sys.executable` and compares the generated sources. A non-zero exit fails
    # the session; the script writes report.txt + diffs/ under the workdir.
    pytest_args = [
        "--cache-clear",
        "-sv",
        "-n",
        "auto",
        "--dist",
        "loadgroup",
        # See the note in 'test_next' above.
        "--require-optional-deps",
        "-m",
        f"{markers}",
        str(pathlib.Path("tests") / "next_tests"),
    ]
    session.run(
        "python",
        str(pathlib.Path("scripts") / "python" / "dace_determinism.py"),
        "ci-check",
        "--workdir",
        str(REPO_ROOT / "_dace_deterministic_codegen"),
        "--",
        *pytest_args,
        *session.posargs,
        env=session.env,
    )


if __name__ == "__main__":
    nox.main()
