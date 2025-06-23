#! /usr/bin/env -S uv run -q --frozen --isolated --group scripts --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage test sessions in GT4Py CI."""

from __future__ import annotations

import enum
import fnmatch
import itertools
import json
import pathlib
import subprocess
from collections.abc import Callable, Iterable, Sequence
from typing import Annotated, Final, TypedDict

from typing_extensions import NotRequired

import rich
import typer
import yaml


REPO_ROOT: Final = pathlib.Path(__file__).parent.parent
DEFAULT_CONFIG = "{REPO_ROOT}/nox-sessions-config.yml"


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    INVALID_COMMAND_OPTIONS = -1
    CONFIG_FILE_NOT_FOUND = -10
    YAML_PARSE_ERROR = -11
    INVALID_SESSION_DEFINITION = -12
    GIT_DIFF_ERROR = -20
    GH_MATRIX_CREATION_ERROR = -30
    NOX_JSON_PARSE_ERROR = -31


class OutputFormat(str, enum.Enum):
    """Output formats for the affected sessions."""

    JSON = "json"
    GH_ACTIONS = "gh-actions"
    GITLAB_CI = "gitlab-ci"

    def __str__(self) -> str:
        return super().__str__()


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


app = typer.Typer(no_args_is_help=True, name="test-sessions", help=__doc__)


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
        out = subprocess.run(cmd_args, capture_output=True, text=True, cwd=REPO_ROOT).stdout
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


def get_nox_sessions(*args: str, verbose: bool = False) -> list[NoxSessionDefinition]:
    """Get the names of nox sessions that should run based on the test sessions configuration."""
    cmd_args = ["./noxfile.py", "--list", "--json", *args]
    try:
        out = subprocess.run(cmd_args, capture_output=True, text=True, cwd=REPO_ROOT).stdout
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


def create_github_actions_list(
    affected_sessions: Sequence[str], *, verbose: bool = False
) -> list[dict[str, str]]:
    """Create a GitHub Actions matrix from the affected CPU-only sessions."""

    nox_sessions = get_nox_sessions("-k", "cpu", verbose=verbose)
    entries: list[dict[str, str]] = []

    processed_sessions = set()
    for session in nox_sessions:
        if session["name"] in affected_sessions:
            call_spec = tuple(session["call_spec"].items())
            session_id = (session["name"], *call_spec)
            if session_id not in processed_sessions:
                processed_sessions.add(session_id)
                entries.append(
                    dict(name=session["name"], args=", ".join(session["call_spec"].values()))
                )

    return entries


def create_gitlab_ci_config(
    affected: Sequence[str], *, base_commit: str = "main", verbose: bool = False
) -> dict[str, object]:
    """Create a GitLab CI configuration for the affected sessions."""
    nox_sessions = get_nox_sessions("-k", "cpu", verbose=verbose)

    gitlab_ci_config = {
        "stages": ["test"],
        "test": {
            "stage": "test",
            "script": [
                f"nox -s {{session}} -- {session['args']}"
                for session in create_github_actions_list(affected, verbose=verbose)
            ],
        },
    }

    return {}


@app.command()
def required(
    *,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Sessions configuration file")
    ] = DEFAULT_CONFIG,
    base_commit: Annotated[str, typer.Option("--base", help="Base commit for changes")] = "main",
    format: Annotated[
        str | None,
        typer.Option("--format", help=f"Output format: {list(f'{f.value}' for f in OutputFormat)}"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", help=f"Output file name."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """List all test sessions affected by changes from a base commit."""

    if format:
        if format not in OutputFormat.__members__.values():
            rich.print(
                f"[red]Error:[/red] Invalid output format: {format!r}. "
                f"Choose from: {list(f'{f.value}' for f in OutputFormat)}"
            )
            raise typer.Exit(ExitCode.INVALID_COMMAND_OPTIONS)

        if not output:
            rich.print(
                "[red]Error:[/red] Output file name must be provided when using --format option."
            )
            raise typer.Exit(ExitCode.INVALID_COMMAND_OPTIONS)

    config_path = config.format(REPO_ROOT=REPO_ROOT)
    sessions = load_test_sessions_config(config_path)
    rich.print(
        f"Found {len(sessions)} test sessions in {config_path}: {[session['name'] for session in sessions]}"
    )

    affected = [
        session["name"]
        for session in sessions
        if should_run_session(session, base_commit=base_commit, verbose=verbose)
    ]

    if format:
        match format:
            case OutputFormat.JSON.value:
                with open(output, "w") as f:
                    json.dump(
                        {"base_commit": base_commit, "affected_sessions": affected}, f, indent=2
                    )
                rich.print(f"Saved affected sessions to {output}")

            case OutputFormat.GH_ACTIONS.value:
                matrix = create_github_actions_list(affected, verbose=verbose)
                with open(output, "w") as f:
                    json.dump(matrix, f, indent=2)
                rich.print(f"Saved GitHub Actions matrix to {output}")

            case OutputFormat.GITLAB_CI.value:
                gitlab_ci_config = create_gitlab_ci_config(affected, base_commit=base_commit)

                with open(output, "w") as f:
                    yaml.dump(gitlab_ci_config, f, default_flow_style=False)
                rich.print(f"Saved GitLab CI configuration to {output}")

            case _:
                assert False, f"Unexpected output format: {format!r}"

    rich.print(
        f"Found {len(affected)} affected test sessions in changes from {base_commit!r}: {affected}"
    )


@app.command()
def all(config_file: Annotated[str, typer.Argument()] = DEFAULT_CONFIG) -> None:
    """List all test sessions defined in the configuration file."""
    config_path = config_file.format(REPO_ROOT=REPO_ROOT)
    sessions = load_test_sessions_config(config_path)

    rich.print(f"Found {len(sessions)} test sessions in {config_path}:")
    for session in sessions:
        session_fields = "\n\t".join(
            ["", *(f"[yellow]{key!s}[/yellow]: {session[key]!r}" for key in session)]
        )
        rich.print(f"- [bold]{session['name']}[/bold]:{session_fields}]\n")


if __name__ == "__main__":
    app()
