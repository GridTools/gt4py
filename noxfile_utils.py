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
import types
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar

import nox
import nox.registry

PROJECT_PREFIX: Final = "GT4PY"

CI_SOURCE_COMMIT_ENV_VAR_NAME: Final = f"{PROJECT_PREFIX}_CI_NOX_RUN_ONLY_IF_CHANGED_FROM"

VERBOSE_MODE: Final = os.environ.get(f"{PROJECT_PREFIX}_CI_NOX_VERBOSE", "").lower() not in [
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
                    f"Skipping session '{session.name}' because it is not relevant for the current changes."
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
            f"  - Environment: {env}\n"
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
    """
    Filter names based on include and exclude `fnmatch`-style patterns.

    Args:
        names: List of names to filter.
        include_patterns: List of patterns to include names.
        exclude_patterns: List of patterns to exclude names.
    Returns:
        A set of names that either match the include patterns and don't match
        the exclude patterns (checked in that order).
    """
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
    commit_spec = os.environ.get(CI_SOURCE_COMMIT_ENV_VAR_NAME, "")
    if not commit_spec:
        return False

    if commit_spec not in _changed_files_from_commit:
        out = session.run(
            *f"git diff --name-only {commit_spec}".split(), external=True, silent=True
        )
        _changed_files_from_commit[commit_spec] = out.strip().split("\n")

    changed_files = _changed_files_from_commit[commit_spec]
    if VERBOSE_MODE:
        print(f"Modified files from '{commit_spec}': {changed_files}")

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
            f"\n[{session.name}]: \n"
        )

    return len(relevant_files) == 0
