# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions and decorators for managing and customizing Nox sessions
in the GT4Py project. It includes functionality for filtering environment
variables, handling session-specific metadata, and determining whether a
session should be skipped based on changes in the repository.

Environment Variables:
- `{PROJECT_PREFIX}_CI_NOX_RUN_ONLY_IF_CHANGED_FROM`: Specifies the commit to compare changes against.
- `{PROJECT_PREFIX}_CI_NOX_VERBOSE`: Enables verbose mode for debugging session behavior.

"""

from __future__ import annotations

import fnmatch
import functools
import itertools
import os
import sys
import types
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeAlias

import nox
import nox.registry

PROJECT_PREFIX: Final = "GT4PY"

CI_SOURCE_COMMIT_ENV_VAR_NAME: Final = f"{PROJECT_PREFIX}_CI_NOX_RUN_ONLY_IF_CHANGED_FROM"

VERBOSE_MODE: Final = os.environ.get(f"{PROJECT_PREFIX}_CI_NOX_VERBOSE", "").lower() in [
    "1",
    "on",
    "true",
]

_changed_files_from_commit: dict[str, list[str]] = {}
_metadata_registry: dict[str, types.SimpleNamespace] = {}


# -- Extra utilities to define test sessions --
AnyCallable: TypeAlias = Callable[..., Any]


def customize_session(
    env_vars: Sequence[str] = (),
    ignore_env_vars: Sequence[str] = (),
    paths: Sequence[str] = (),
    ignore_paths: Sequence[str] = (),
    **kwargs: Any,
) -> Callable[[AnyCallable], nox.registry.Func]:
    """Customize a Nox session with path or environment-based filtering.

    Define a Nox session with the ability to sandbox the environment variables
    that are passed to the test environment and to conditionally skip the session
    based on the paths of the modified files.

    Args:
        env_vars: A sequence of environment variable names that will be passed
            from the outer environment to the test environment.
        ignore_env_vars: A sequence of environment variable names that should be
            removed from the outer environment before passing to the test environment.
        paths: A sequence of file paths or patterns that when changed will trigger the session.
        ignore_paths: A sequence of file paths or patterns that when changed will cause the session to be skipped.
            **kwargs: Additional keyword arguments passed to `nox.session` decorator.

    Note:
        Only one of the accept/ignore argument pairs can be used at the same time.

    Example:
        ```python
        @customize_session(paths=["src/", "tests/"], name="test")
        def run_tests(session):
            session.run("pytest")
        ```
    """

    if env_vars and ignore_env_vars:
        raise ValueError("Cannot use both `env_vars` and `ignore_env_vars` at the same time.")
    if paths and ignore_paths:
        raise ValueError("Cannot use both `paths` and `ignore_paths` at the same time.")

    def decorator(session_function: AnyCallable) -> nox.registry.Func:
        assert (
            session_function.__name__ not in _metadata_registry
        ), f"Session function '{session_function.__name__}' already has metadata."
        _metadata_registry[kwargs.get("name", session_function.__name__)] = types.SimpleNamespace(
            env_vars=env_vars,
            ignore_env_vars=ignore_env_vars,
            paths=paths,
            ignore_paths=ignore_paths,
        )

        @functools.wraps(session_function)
        def new_session_function(*args, **kwargs) -> Any:
            session = kwargs.get("session", None) or args[0]
            if _is_skippable_session(session):
                print(
                    f"Skipping session '{session.name}' because it is not relevant for the current changes.",
                    file=sys.stderr,
                )
            else:
                session_function(*args, **kwargs)

        for key, value in vars(session_function).items():
            setattr(new_session_function, key, value)

        return nox.session(**kwargs)(new_session_function)

    return decorator


def install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""

    unversioned_session_name = session.name.split("-")[0]
    metadata = _metadata_registry[unversioned_session_name]
    env = {
        key: os.environ.get(key)
        for key in _filter_names(os.environ.keys(), metadata.env_vars, metadata.ignore_env_vars)
    } | {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    if VERBOSE_MODE:
        print(
            f"\n[{session.name}]:\n"
            f"  - Allow env variables patterns: {metadata.env_vars}\n"
            f"  - Ignore env variables patterns: {metadata.ignore_env_vars}\n"
            f"\n[{session.name}]:\n"
            f"  - Environment: {env}\n",
            file=sys.stderr,
        )

    session.run_install(
        "uv",
        "sync",
        *("--python", session.python),
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env=env,
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        )


# -- Internal implementation utilities --
def _filter_names(
    names: list[str], include_patterns: list[str], exclude_patterns: list[str]
) -> str:
    """Filter names based on include and exclude `fnmatch`-style patterns."""
    included = (
        set(
            itertools.chain(
                *(fnmatch.filter(names, include_pattern) for include_pattern in include_patterns)
            )
        )
        if include_patterns
        else set(names)
    )

    excluded = (
        set(
            itertools.chain(
                *(fnmatch.filter(included, exclude_pattern) for exclude_pattern in exclude_patterns)
            )
        )
        if exclude_patterns
        else set()
    )

    return included - excluded


def _is_skippable_session(session: nox.Session) -> None:
    """Determine if a session can be skipped based on changed files from a specific commit.

    This function checks if a session can be skipped by analyzing which files have changed
    from a specific commit and comparing them against the session's relevant paths.

    Notes:
        - Uses environment variables to get the commit target and the verbose mode.
        - Uses git diff to find changed files from the commit specified in the environment variable.
        - When VERBOSE_MODE is enabled, prints detailed information about the decision process.
    """
    commit_spec = os.environ.get(CI_SOURCE_COMMIT_ENV_VAR_NAME, "")
    if not commit_spec:
        return False

    if commit_spec not in _changed_files_from_commit:
        out = session.run(
            *f"git diff --name-only {commit_spec} --".split(), external=True, silent=True
        )
        _changed_files_from_commit[commit_spec] = out.strip().split("\n")

    changed_files = _changed_files_from_commit[commit_spec]
    if VERBOSE_MODE:
        print(f"Modified files from '{commit_spec}': {changed_files}", file=sys.stderr)

    unversioned_session_name = session.name.split("-")[0]
    metadata = _metadata_registry[unversioned_session_name]

    relevant_files = _filter_names(changed_files, metadata.paths, metadata.ignore_paths)
    if VERBOSE_MODE:
        print(
            f"\n[{session.name}]:\n"
            f"  - File include patterns: {metadata.paths}\n"
            f"  - File exclude patterns: {metadata.ignore_paths}\n"
            f"  - Changed files: {list(changed_files)}\n"
            f"  - Relevant files: {list(relevant_files)}\n"
            f"\n[{session.name}]: \n",
            file=sys.stderr,
        )

    return len(relevant_files) == 0
