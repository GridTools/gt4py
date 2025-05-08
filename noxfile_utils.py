# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions and decorators for managing and customizing Nox sessions.

It includes functionality for handling session-specific metadata, like determining
whether a session should be skipped based on the changes in the repository.

Environment Variables:

- `[{PROJECT_PREFIX}_]CI_NOX_RUN_ONLY_IF_CHANGED_FROM`: Specifies the
  commit to compare changes against.
- `[{PROJECT_PREFIX}_]CI_NOX_VERBOSE`: Enables verbose mode for debugging
  session behavior.

"""

from __future__ import annotations

import fnmatch
import functools
import itertools
import os
import pathlib
import subprocess
import sys
import types
import unittest
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, TypeAlias
import unittest.mock

import nox
from nox._decorators import Func as NoxFunc

ENV_VAR_PREFIX: Final = f"{prefix}_" if (prefix := os.environ.get("CI_NOX_PREFIX", "")) else ""
ENV_VAR_RUN_ONLY_IF_CHANGED_FROM: Final = f"{ENV_VAR_PREFIX}CI_NOX_RUN_ONLY_IF_CHANGED_FROM"
ENV_VAR_VERBOSE: Final = f"{ENV_VAR_PREFIX}CI_NOX_VERBOSE"

TARGET_GIT_SPEC_FOR_CHANGES: Final = os.environ.get(ENV_VAR_RUN_ONLY_IF_CHANGED_FROM, "")
VERBOSE: Final = os.environ.get(ENV_VAR_VERBOSE, "").lower() in [
    "1",
    "on",
    "true",
]

_metadata_registry: dict[str, types.SimpleNamespace] = {}


# -- Extra utilities to define test sessions --
AnyCallable: TypeAlias = Callable[..., Any]


def customize_session(
    paths: Sequence[str] = (),
    ignore_paths: Sequence[str] = (),
    **kwargs: Any,
) -> Callable[[AnyCallable], NoxFunc]:
    """
    Customize a Nox session with path-based filtering.

    Define a Nox session with the ability to conditionally skip the session
    based on the paths of the modified files.

    Args:
        paths: A sequence of file paths or patterns that when changed will trigger the session.
        ignore_paths: A sequence of file paths or patterns that when changed will cause the
            session to be skipped.
        **kwargs: Additional keyword arguments passed to `nox.session` decorator.

    Note:
        Only one of the `paths` / `ignore_paths` options can be used at the same time for now.

    Example:
        ```python
        @customize_session(paths=["src/", "tests/"], name="test")
        def run_tests(session):
            session.run("pytest")
        ```
    """
    if paths and ignore_paths:
        raise ValueError("Cannot use both 'paths' and 'ignore_paths' at the same time.")

    def decorator(session_function: AnyCallable) -> NoxFunc:
        session_name = kwargs.get("name", session_function.__name__)
        _metadata_registry.setdefault(session_name, types.SimpleNamespace()).__dict__.update(
            paths=paths,
            ignore_paths=ignore_paths,
        )

        @functools.wraps(session_function)
        def new_session_function(*args: Any, **kwargs: Any) -> Any:
            session = kwargs.get("session", None) or args[0]
            if is_required_by_repo_changes(session.name):
                session_function(*args, **kwargs)
            else:
                print(
                    f"Skipping session '{session.name}' because it is not relevant for the current changes.",
                    file=sys.stderr,
                )

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
        str(session.python),
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


def is_required_by_repo_changes(
    session: nox.Session | str,
    target_git_spec: str = TARGET_GIT_SPEC_FOR_CHANGES,
    *,
    verbose: bool = VERBOSE,
) -> bool:
    """
    Determine if a session is affected by changes from a commit in the git repository.

    The decision is taken based on the session's registered file paths and ignore patterns.

    Args:
        session:
            The nox session to check or a string representing the session name.
        target_git_spec:
            The git commit specification to use for determining changes (e.g., 'HEAD~1..HEAD').
            If empty, the function returns True, indicating all sessions are required.
        verbose:
            Whether to print detailed information about the affected files.

    Returns:
        True if the session is affected by the repository changes, False otherwise.

    Notes:
        - The function caches the list of changed files for each commit specification to avoid
        multiple git calls.
        - For a session to be affected, at least one changed file must match the session's
        registered paths and not be in the ignore_paths.
        - Session metadata (paths and ignore_paths) is retrieved from the _metadata_registry.
    """
    if not target_git_spec:
        return True

    cmd_args = ["git", "diff", "--name-only", target_git_spec]
    if isinstance(session, str):
        cwd = pathlib.Path(__file__).parent
        out = subprocess.run(cmd_args, capture_output=True, text=True, cwd=cwd).stdout
    else:
        out = session.run(*cmd_args, external=True, silent=True)  # type: ignore[assignment]
    changed_files = out.strip().split("\n")

    session_name: str = getattr(session, "name", session)  # type: ignore[arg-type]
    unversioned_session_name = session_name.split("-")[0]
    metadata = _metadata_registry.get(unversioned_session_name, None)
    paths = metadata.paths if metadata else ()
    ignore_paths = metadata.ignore_paths if metadata else ()

    relevant_files = _filter_names(changed_files, paths, ignore_paths)
    is_affected = len(relevant_files) > 0
    if verbose:
        print(
            f"\n[{session_name}]:\n"
            f"  - Required: {is_affected}\n"
            f"  - Target git spec: {target_git_spec!r}\n"
            f"  - File include patterns: {paths}\n"
            f"  - File exclude patterns: {ignore_paths}\n"
            f"  - Relevant files ({len(relevant_files)}/{len(changed_files)}):\n",
            "\n".join(f"\t+ [{'x' if f in relevant_files else ' '}] {f}" for f in changed_files),
            "\n",
            file=sys.stderr,
        )

    return is_affected


# -- Internal implementation utilities --
def _filter_names(
    names: Iterable[str],
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
) -> list[str]:
    """Filter names based on include and exclude `fnmatch`-style patterns."""

    def _filter(names: Iterable[str], patterns: Iterable[str]) -> Iterable[str]:
        return itertools.chain(*(fnmatch.filter(names, pattern) for pattern in patterns))

    included = set(_filter(names, include_patterns) if include_patterns else names)
    excluded = set(_filter(included, exclude_patterns) if exclude_patterns else [])

    return sorted(included - excluded)


# -- Self unit tests --
class NoxUtilsTestCase(unittest.TestCase):
    def test_filter_names(self):
        names = ["foo", "bar", "baz"]

        include_patterns = ["f*", "b*"]
        exclude_patterns = ["b*"]
        self.assertEqual(_filter_names(names, include_patterns, exclude_patterns), ["foo"])

        include_patterns = ["*a*"]
        exclude_patterns = ["*r"]
        self.assertEqual(_filter_names(names, include_patterns, exclude_patterns), ["baz"])

        include_patterns = None
        exclude_patterns = ["*o"]
        self.assertEqual(_filter_names(names, include_patterns, exclude_patterns), ["bar", "baz"])

        include_patterns = ["b*"]
        exclude_patterns = None
        self.assertEqual(_filter_names(names, include_patterns, exclude_patterns), ["bar", "baz"])

    def test_is_affected_by_repo_changes(self):
        _metadata_registry["test_session"] = types.SimpleNamespace(
            paths=["src/*"], ignore_paths=["tests/*"]
        )

        # Source commit defined
        commit_spec = "main"
        session = unittest.mock.Mock()
        session.name = "test_session-3.10(param1=foo,param2=bar)"

        session.run.return_value = "src/foo.py"
        self.assertTrue(is_required_by_repo_changes(session, commit_spec))

        session.run.return_value = "src/foo.py\ntests/test_foo.py"
        self.assertTrue(is_required_by_repo_changes(session, commit_spec))

        session.run.return_value = "tests/test_foo.py"
        self.assertFalse(is_required_by_repo_changes(session, commit_spec))

        session.run.return_value = "docs/readme.md"
        self.assertFalse(is_required_by_repo_changes(session, commit_spec))

        session.run.return_value = "src/bar.py\nsrc/baz.py"
        self.assertTrue(is_required_by_repo_changes(session, commit_spec))

        # Undefined source commit
        with unittest.mock.patch.dict(os.environ, {}):
            session = unittest.mock.Mock()
            session.name = "test_session-3.10(param1=foo,param2=bar)"

            self.assertTrue(is_required_by_repo_changes(session))
            session.run.assert_not_called()


# Run this file as a script to execute mypy checks and  unit tests.
if __name__ == "__main__":
    unittest.main()
