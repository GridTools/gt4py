#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

#! /usr/bin/env -S uv run -q -p 3.11 --frozen --isolated --group scripts --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage conditional execution of nox sessions."""

from __future__ import annotations

import enum
import fnmatch
import itertools
import json
import pathlib
import subprocess
from collections.abc import Iterable, Sequence
from typing import Annotated, Final, NotRequired, TypedDict

import rich
import typer
import yaml

from . import _common as common


DEFAULT_CONFIG: Final = "{REPO_ROOT}/nox-sessions-config.yml"


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    INVALID_COMMAND_OPTIONS = 1
    CONFIG_FILE_NOT_FOUND = 10
    YAML_PARSE_ERROR = 11
    INVALID_SESSION_DEFINITION = 12
    GIT_DIFF_ERROR = 20
    GH_MATRIX_CREATION_ERROR = 30
    NOX_JSON_PARSE_ERROR = 31


SessionDefinition = TypedDict(
    "SessionDefinition",
    {"name": str, "paths": NotRequired[list[str]], "ignore-paths": NotRequired[list[str]]},
)

NoxSessionDefinition = TypedDict(
    "NoxSessionDefinition",
    {
        "session": str,
        "name": str,
        "description": str,
        "python": str,
        "tags": list[str],
        "call_spec": dict[str, str],
    },
)


cli = typer.Typer(no_args_is_help=True, name="nox-sessions", help=__doc__)


def load_test_sessions_config(config_path: str | pathlib.Path) -> list[SessionDefinition]:
    """Load and parse the yaml file with the test sessions definitions."""
    sessions = []
    try:
        with open(config_path) as f:
            sessions_file = yaml.safe_load(f)

        sessions = sessions_file.get("sessions", [])

    except FileNotFoundError as e:
        rich.print(f"Configuration file not found: {config_path}")
        raise typer.Exit(ExitCode.CONFIG_FILE_NOT_FOUND) from e

    except yaml.YAMLError as e:
        rich.print(f"Error parsing YAML configuration file {config_path}: {e}")
        raise typer.Exit(ExitCode.YAML_PARSE_ERROR) from e

    except AttributeError as e:
        rich.print(f"Invalid session definition in {config_path}: {e}")
        raise typer.Exit(ExitCode.INVALID_SESSION_DEFINITION) from e

    return sessions


def get_changed_files(base_commit: str) -> list[str]:
    """Get list of changed files from base_commit."""
    cmd_args = ["git", "diff", "--name-only", base_commit]
    try:
        out = subprocess.run(cmd_args, capture_output=True, text=True, cwd=common.REPO_ROOT).stdout
    except subprocess.CalledProcessError as e:
        rich.print(f"[red]Error:[/red] Failed to get changed files: {e}")
        raise typer.Exit(ExitCode.GIT_DIFF_ERROR) from e

    changed_files = out.strip().split("\n")

    return changed_files


def filter_names(
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


def should_run_session(
    session: SessionDefinition, *, base_commit: str = "main", verbose: bool = False
) -> bool:
    """Determine if a session should run based on the changes from a base commit."""

    changed_files = get_changed_files(base_commit)
    paths = session.get("paths", [])
    ignore_paths = session.get("ignore-paths", [])
    relevant_files = filter_names(changed_files, paths, ignore_paths)
    should_run = len(relevant_files) > 0

    if verbose:
        file_list = []
        for f in changed_files:
            if f in relevant_files:
                file_list.append(f"\t[green][bold]+ {f}[/bold][/green]")
            else:
                file_list.append(f"\t- {f}")

        rich.print(
            "\n".join(
                [
                    f"[green]{session['name']}[/green]:"
                    if should_run
                    else f"[red]{session['name']}[/red]:",
                    f"  - File include patterns: {paths}",
                    f"  - File exclude patterns: {ignore_paths}",
                    f"  - Relevant files ({len(relevant_files)}/{len(changed_files)}):",
                    *file_list,
                    "\n",
                ]
            )
        )

    return should_run


def get_sessions(*args: str, verbose: bool = False) -> list[NoxSessionDefinition]:
    """Get the names of nox sessions that should run based on the test sessions configuration."""
    cmd_args = ["./noxfile.py", "--list", "--json", *args]
    try:
        out = subprocess.run(cmd_args, capture_output=True, text=True, cwd=common.REPO_ROOT).stdout
        if verbose:
            rich.print(f"nox output: {out}")
        nox_sessions = json.loads(out)
    except subprocess.CalledProcessError as e:
        rich.print(f"[red]Error:[/red] Failed to get test session info from nox: {e}")
        raise typer.Exit(ExitCode.GH_MATRIX_CREATION_ERROR) from e
    except json.JSONDecodeError as e:
        rich.print(f"[red]Error:[/red] Failed to parse nox output as JSON: {e}")
        rich.print("[red]Hint:[/red] Make sure you have nox installed and the noxfile is valid.")
        raise typer.Exit(ExitCode.NOX_JSON_PARSE_ERROR) from e

    return nox_sessions


@cli.command()
def required(
    *,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Sessions configuration file")
    ] = DEFAULT_CONFIG,
    base_commit: Annotated[str, typer.Option("--base", help="Base commit for changes")] = "main",
    json_output: Annotated[
        str | None, typer.Option("--json-output", help="Output file name.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """List all test sessions affected by changes from a base commit."""

    config_path = config.format(REPO_ROOT=common.REPO_ROOT)
    sessions = load_test_sessions_config(config_path)
    rich.print(
        f"Found {len(sessions)} test sessions in {config_path}: {[session['name'] for session in sessions]}"
    )

    affected = [
        session["name"]
        for session in sessions
        if should_run_session(session, base_commit=base_commit, verbose=verbose)
    ]

    if json_output:
        with open(json_output, "w") as f:
            json.dump({"base_commit": base_commit, "affected_sessions": affected}, f, indent=2)
        rich.print(f"Saved affected sessions to {json_output}")

    rich.print(
        f"Found {len(affected)} affected test sessions in changes from {base_commit!r}: {affected}"
    )


@cli.command()
def list(config_file: Annotated[str, typer.Argument()] = DEFAULT_CONFIG) -> None:  # noqa: A001 [builtin-variable-shadowing]
    """List all test sessions defined in the configuration file."""
    config_path = config_file.format(REPO_ROOT=common.REPO_ROOT)
    sessions = load_test_sessions_config(config_path)

    rich.print(f"Found {len(sessions)} test sessions in {config_path}:")
    for session in sessions:
        session_fields = "\n\t".join(
            ["", *(f"[yellow]{key!s}[/yellow]: {session[key]!r}" for key in session)]
        )
        rich.print(f"- [bold]{session['name']}[/bold]:{session_fields}]\n")


if __name__ == "__main__":
    cli()
