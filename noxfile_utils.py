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
import unittest
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, TypeAlias
import unittest.mock

import nox
import nox.registry

ENV_VAR_PREFIX: Final = f"{prefix}_" if (prefix := os.environ.get("CI_NOX_PREFIX", "")) else ""
ENV_VAR_RUN_ONLY_IF_CHANGED_FROM: Final = f"{ENV_VAR_PREFIX}CI_NOX_RUN_ONLY_IF_CHANGED_FROM"
ENV_VAR_VERBOSE: Final = f"{ENV_VAR_PREFIX}CI_NOX_VERBOSE"

VERBOSE: Final = os.environ.get(ENV_VAR_VERBOSE, "").lower() in [
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
        raise ValueError("Cannot use both 'env_vars' and 'ignore_env_vars' at the same time.")
    if paths and ignore_paths:
        raise ValueError("Cannot use both 'paths' and 'ignore_paths' at the same time.")

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
    """
    Install session packages using the `uv` tool.

    Args:
        session: The Nox session object.
        *args: Additional packages to install in the session (via `uv pip install`)
        extras: Names of package's extras to install.
        groups: Names of dependency groups to install.
    """

    env = make_session_env(session, UV_PROJECT_ENVIRONMENT=session.virtualenv.location)
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
            env=env,
        )


def make_session_env(session: nox.Session, **kwargs: str) -> dict[str, str]:
    """Create an environment dictionary for a nox session.

    This function builds an environment dictionary for a nox session based on registered metadata.
    It filters environment variables according to allowed and ignored patterns defined in the
    session metadata, and combines them with any additional key-value pairs provided.

    Args:
        session: The nox session for which to create the environment.
        **kwargs: Additional environment variables to include in the returned dictionary.

    Returns:
        A dictionary containing the filtered environment variables from the current
        environment, combined with any provided kwargs.
    """
    unversioned_session_name = session.name.split("-")[0]
    metadata = _metadata_registry.get(unversioned_session_name, None)
    env_vars = metadata.env_vars if metadata else ()
    ignore_env_vars = metadata.ignore_env_vars if metadata else ()

    env = {
        key: os.environ.get(key)
        for key in _filter_names(os.environ.keys(), env_vars, ignore_env_vars)
    } | kwargs

    if VERBOSE:
        print(
            f"\n[{session.name}]:\n"
            f"  - Allow env variables patterns: {env_vars}\n"
            f"  - Ignore env variables patterns: {ignore_env_vars}\n"
            f"\n[{session.name}]:\n"
            f"  - Environment: {env}\n",
            file=sys.stderr,
        )

    return env


def run_session(session: nox.Session, *args: str | Sequence[str], **kwargs: Any) -> None:
    """Run a Nox session with the specified arguments and environment.

    This function runs a Nox session with the provided arguments, using the
    environment variables defined in the session metadata.

    Args:
        session: The Nox session to run.
        *args: Additional arguments to pass to the session.
        **kwargs: Additional keyword arguments to pass to the session.
    """
    env = make_session_env(session)
    return session.run(*args, **kwargs, env=env)


# -- Internal implementation utilities --
def _filter_names(
    names: Iterable[str],
    include_patterns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
) -> list[str]:
    """Filter names based on include and exclude `fnmatch`-style patterns."""

    def _filter(names: Iterable[str], patterns: Iterable[str]) -> set[str]:
        return itertools.chain(*(fnmatch.filter(names, pattern) for pattern in patterns))

    included = set(_filter(names, include_patterns) if include_patterns else names)
    excluded = set(_filter(included, exclude_patterns) if exclude_patterns else [])

    return list(sorted(included - excluded))


def _is_skippable_session(session: nox.Session) -> bool:
    """Determine if a session can be skipped based on changed files from a specific commit.

    This function checks if a session can be skipped by analyzing which files have changed
    from a specific commit and comparing them against the session's relevant paths.

    Notes:
        - Uses environment variables to get the commit target and the verbose mode.
        - Uses git diff to find changed files from the commit specified in the environment variable.
        - When VERBOSE is enabled, prints detailed information about the decision process.
    """
    commit_spec = os.environ.get(ENV_VAR_RUN_ONLY_IF_CHANGED_FROM, "")
    if not commit_spec:
        return False

    if commit_spec not in _changed_files_from_commit:
        out = session.run(
            *f"git diff --name-only {commit_spec} --".split(), external=True, silent=True
        )
        _changed_files_from_commit[commit_spec] = out.strip().split("\n")

    changed_files = _changed_files_from_commit[commit_spec]
    if VERBOSE:
        print(f"Modified files from '{commit_spec}': {changed_files}", file=sys.stderr)

    unversioned_session_name = session.name.split("-")[0]
    metadata = _metadata_registry.get(unversioned_session_name, None)
    paths = metadata.paths if metadata else ()
    ignore_paths = metadata.ignore_paths if metadata else ()

    relevant_files = _filter_names(changed_files, paths, ignore_paths)
    if VERBOSE:
        print(
            f"\n[{session.name}]:\n"
            f"  - File include patterns: {paths}\n"
            f"  - File exclude patterns: {ignore_paths}\n"
            f"  - Changed files: {list(changed_files)}\n"
            f"  - Relevant files: {list(relevant_files)}\n"
            f"\n[{session.name}]: \n",
            file=sys.stderr,
        )

    return len(relevant_files) == 0


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

    def test_is_skippable_session(self):
        _metadata_registry["test_session"] = types.SimpleNamespace(
            paths=["src/*"], ignore_paths=["tests/*"]
        )

        # Source commit defined, `git diff` already cached
        with unittest.mock.patch.dict(os.environ, {ENV_VAR_RUN_ONLY_IF_CHANGED_FROM: "main"}):
            session = unittest.mock.Mock()
            session.name = "test_session-3.10(param1=foo,param2=bar)"

            _changed_files_from_commit["main"] = ["src/foo.py"]
            self.assertFalse(_is_skippable_session(session))

            _changed_files_from_commit["main"] = ["src/foo.py", "tests/test_foo.py"]
            self.assertFalse(_is_skippable_session(session))

            _changed_files_from_commit["main"] = ["tests/test_foo.py"]
            self.assertTrue(_is_skippable_session(session))

            _changed_files_from_commit["main"] = ["docs/readme.md"]
            self.assertTrue(_is_skippable_session(session))

            # No need to run `git diff`, already cached
            session.run.assert_not_called()

        # Source commit defined, `git diff` not cached
        with unittest.mock.patch.dict(
            os.environ, {ENV_VAR_RUN_ONLY_IF_CHANGED_FROM: "not_cached_commit"}
        ):
            session = unittest.mock.Mock(run=lambda *args, **kwargs: "src/bar.py\nsrc/baz.py")
            session.name = "test_session-3.10(param1=foo,param2=bar)"

            self.assertFalse(_is_skippable_session(session))
            self.assertEqual(
                _changed_files_from_commit["not_cached_commit"], ["src/bar.py", "src/baz.py"]
            )

        # Undefined source commit
        with unittest.mock.patch.dict(os.environ, {}):
            session = unittest.mock.Mock()
            session.name = "test_session-3.10(param1=foo,param2=bar)"

            self.assertFalse(_is_skippable_session(session))
            session.run.assert_not_called()


# Run this file as a script to execute the tests.
if __name__ == "__main__":
    unittest.main()
