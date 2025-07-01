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

import sys
from collections.abc import Sequence
from typing import Annotated

import rich
import typer
import yaml

from . import _common as common, nox_sessions


cli = typer.Typer(no_args_is_help=True, name="cscs-ci", help=__doc__)


def create_gitlab_ci_matrix(
    affected_sessions: Sequence[str], *, verbose: bool = False
) -> list[dict[str, str | list[str]]]:
    """Create a GitLab CI configuration for the affected sessions."""

    sessions = nox_sessions.get_sessions(verbose=verbose)
    entries: list[dict[str, str | list[str]]] = []

    processed_sessions = set()
    for session in sessions:
        if session["name"] in affected_sessions:
            call_spec = tuple(session["call_spec"].items())
            session_id = (session["name"], *call_spec)
            if session_id not in processed_sessions:
                processed_sessions.add(session_id)
                entries.append(
                    dict(
                        SESSION_NAME=session["name"],
                        SESSION_ARGS=", ".join(session["call_spec"].values()),
                        SESSION_PYTHON=[*common.PYTHON_VERSIONS],
                    )
                )

    return entries


@cli.command()
def matrix(
    *,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Sessions configuration file")
    ] = nox_sessions.DEFAULT_CONFIG,
    base_commit: Annotated[str, typer.Option("--base", help="Base commit for changes")] = "main",
    output: Annotated[str | None, typer.Option("--output", help="Output (YAML) file name.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Define matrix of test sessions affected by changes from a base commit."""

    config_path = config.format(REPO_ROOT=common.REPO_ROOT)
    sessions = nox_sessions.load_test_sessions_config(config_path)
    rich.print(
        f"Found {len(sessions)} test sessions in {config_path}: {[session['name'] for session in sessions]}"
    )

    affected = [
        session["name"]
        for session in sessions
        if nox_sessions.should_run_session(session, base_commit=base_commit, verbose=verbose)
    ]

    matrix = create_gitlab_ci_matrix(affected, verbose=verbose)
    if output:
        matrix = create_gitlab_ci_matrix(affected, verbose=verbose)

        with open(output, "w") as f:
            yaml.dump(matrix, f, default_flow_style=False)
        rich.print(f"Saved GitLab CI matrix to '{output}'")
    else:
        rich.print(
            "GitLab CI matrix: (use '--output <filename>' to save it)\n---------------------"
        )
        yaml.dump(matrix, sys.stdout, indent=2, sort_keys=False)

    rich.print(
        f"Found {len(affected)} affected test sessions in changes from {base_commit!r}: {affected}"
    )


if __name__ == "__main__":
    cli()
