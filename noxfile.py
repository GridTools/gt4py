# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import fnmatch
import functools
import itertools
import os
import pathlib
import types
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar

import nox

#: This should just be `pytest.ExitCode.NO_TESTS_COLLECTED` but `pytest`
#: is not guaranteed to be available in the venv where `nox` is running.
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

# -- Extra utilities to define test sessions --
T = TypeVar("T")


def session_metadata(**kwargs: Any) -> Callable[[T], T]:
    """Decorator to add metadata to a nox session.

    This decorator attaches a `_metadata_` attribute to a nox session function
    with the provided keyword arguments.

    Args:
        **kwargs: Arbitrary keyword arguments that will be stored as metadata.
            At least one keyword argument must be provided.

    """
    assert kwargs

    def decorator(session_object: T) -> T:
        assert not hasattr(session_object, "_metadata_")
        session_object._metadata_ = kwargs
        return session_object

    return decorator


# -- Sessions --
@nox.session(python=["3.10", "3.11"], tags=["ci"])
def ci_against_commit(session: nox.Session) -> None:
    import optparse

    opt_parser = optparse.OptionParser(
        usage="\n".join(
            [
                "nox -e ci_against_commit -- [OPTIONS] [SESSION_NAMES]",
                "",
                "Example: nox -e ci_against_commit -- -c main test_cartesian",
            ]
        ),
        option_list=[
            optparse.make_option("-c", "--commit", action="store", default="main", dest="commit"),
            optparse.make_option(
                "-v", "--verbose", action="store_true", default=False, dest="verbose"
            ),
        ],
    )
    options, args = opt_parser.parse_args(session.posargs)

    out = session.run(*f"git diff --name-only {options.commit}".split(), external=True, silent=True)
    changed_files = out.strip().split("\n")
    if options.verbose:
        print(f"Modified files from '{options.commit}': {changed_files}")

    for bare_session_name in args:
        test_session = globals()[bare_session_name]
        paths = []
        ignore_paths = []
        if hasattr(test_session, "_metadata_"):
            assert (
                len({"paths", "ignore_paths"}.intersection(test_session._metadata_.keys())) < 2
            ), "Only one of 'include_paths' or 'exclude_paths' can be specified."
            paths = test_session._metadata_.get("paths", [])
            ignore_paths = test_session._metadata_.get("ignore_paths", [])

        session_name = f"{bare_session_name}-{session.python}"
        relevant_files = _filter_filenames(changed_files, paths, ignore_paths)
        if options.verbose:
            print(
                f"\n'{session_name}':\n"
                f"  - Include patterns: {paths}\n"
                f"  - Exclude patterns: {ignore_paths}\n"
                f"  - Relevant files: {list(relevant_files)}\n"
            )
        if relevant_files:
            print(f"'{session_name}': Running tests")
            print(f"session.notify({session_name})")
        else:
            print(f"'{session_name}': Skipping")


@session_metadata(
    ignore_paths=["src/gt4py/next/*", "tests/next_tests/**", "examples/**", "*.md", "*.rst"]
)
@nox.session(python=["3.10", "3.11"], tags=["cartesian"])
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

    _install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=["test"],
    )

    num_processes = os.environ.get("NUM_PROCESSES", "auto")
    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"])

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes} --dist loadgroup".split(),
        *("-m", f"{markers}"),
        str(pathlib.Path("tests") / "cartesian_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules --doctest-ignore-import-errors -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "cartesian"),
    )


@session_metadata(  # Run when gt4py.eve files (or package settings) are changed
    paths=[
        "src/gt4py/eve/*",
        "tests/eve_tests/*",
        ".github/workflows/*",
        "*.lock",
        "*.toml",
        "*.yml",
        "noxfile.py",
    ]
)
@nox.session(python=["3.10", "3.11"], tags=["cartesian", "next", "cpu"])
def test_eve(session: nox.Session) -> None:
    """Run 'gt4py.eve' tests."""

    _install_session_venv(session, groups=["test"])

    num_processes = os.environ.get("NUM_PROCESSES", "auto")

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
        str(pathlib.Path("tests") / "eve_tests"),
        *session.posargs,
    )
    session.run(
        *"pytest --doctest-modules -sv".split(),
        str(pathlib.Path("src") / "gt4py" / "eve"),
    )


@nox.session(python=["3.10", "3.11"])
def test_examples(session: nox.Session) -> None:
    """Run and test documentation workflows."""

    _install_session_venv(session, extras=["testing"], groups=["docs", "test"])

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


@session_metadata(  # Skip when only gt4py.cartesian or doc files have been updated
    ignore_paths=[
        "src/gt4py/cartesian/**",
        "tests/cartesian_tests/**",
        "examples/**",
        "*.md",
        "*.rst",
    ]
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

    _install_session_venv(
        session,
        extras=["performance", "testing", *codegen_settings["extras"], *device_settings["extras"]],
        groups=groups,
    )

    num_processes = os.environ.get("NUM_PROCESSES", "auto")
    markers = " and ".join(codegen_settings["markers"] + device_settings["markers"] + mesh_markers)

    session.run(
        *f"pytest --cache-clear -sv -n {num_processes}".split(),
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


@nox.session(python=["3.10", "3.11"], tags=["cartesian", "next", "cpu"])
def test_package(session: nox.Session) -> None:
    """Run 'gt4py' package level tests."""

    _install_session_venv(session, groups=["test"])

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


@session_metadata(  # Run when gt4py.storage files (or package settings) are changed
    paths=[
        "src/gt4py/storage/**",
        "src/gt4py/cartesian/backend/**",  # For DaCe storages
        "tests/storage_tests/**",
        ".github/workflows/**",
        "*.lock" "*.toml",
        "*.yml",
        "noxfile.py",
    ]
)
@nox.session(python=["3.10", "3.11"], tags=["cartesian", "next"])
@nox.parametrize("device", [DeviceNoxParam.cpu, DeviceNoxParam.cuda12])
def test_storage(
    session: nox.Session,
    device: DeviceOption,
) -> None:
    """Run selected 'gt4py.storage' tests."""

    device_settings = DeviceTestSettings[device]

    _install_session_venv(
        session, extras=["performance", "testing", *device_settings["extras"]], groups=["test"]
    )

    num_processes = os.environ.get("NUM_PROCESSES", "auto")
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
        success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
    )


# -- Internal implementation utilities --
def _filter_filenames(
    filenames: list[str], include_patterns: list[str], exclude_patterns: list[str]
) -> str:
    """
    Filter filenames based on include and exclude patterns.

    Args:
        filenames: List of filenames to filter.
        include_patterns: List of `fnmatch`-style patterns to include files.
        exclude_patterns: List of `fnmatch`-style patterns to exclude files.
    Returns:
        A set of filenames that match the include patterns but not the exclude patterns.
    """
    if include_patterns:
        candidates = set(
            itertools.chain(
                *(
                    fnmatch.filter(filenames, include_pattern)
                    for include_pattern in include_patterns
                )
            )
        )
    else:
        candidates = set(filenames)

    excluded = set(
        itertools.chain(
            *(fnmatch.filter(candidates, exclude_pattern) for exclude_pattern in exclude_patterns)
        )
    )

    return candidates - excluded


def _install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""
    session.run_install(
        "uv",
        "sync",
        *("--python", session.python),
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env={key: value for key, value in os.environ.items()}
        | {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        )
