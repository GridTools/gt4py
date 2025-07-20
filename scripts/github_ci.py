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

import json
from collections.abc import Sequence
from typing import Annotated, Literal, cast

import rich
import typer
from rich import pretty

from . import _common as common, nox_sessions


cli = typer.Typer(no_args_is_help=True, name="github-ci", help=__doc__)


def collect_affected_sessions(
    *,
    config: str = nox_sessions.DEFAULT_CONFIG,
    base_commit: str | None = None,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Collect affected test sessions based on changes from a base commit."""

    config_path = config.format(REPO_ROOT=common.REPO_ROOT)
    sessions = nox_sessions.load_test_sessions_config(config_path)
    if verbose:
        rich.print(
            f"Found {len(sessions)} test sessions in {config_path}: {[session['name'] for session in sessions]}"
        )

    collect_all = base_commit is None
    all_session_names = [session["name"] for session in sessions]
    affected = (
        all_session_names
        if collect_all
        else [
            session["name"]
            for session in sessions
            if nox_sessions.should_run_session(
                session, base_commit=cast(str, base_commit), verbose=verbose
            )
        ]
    )

    if verbose:
        rich.print(
            f"Found {len(affected)} test sessions in changes from {base_commit!r}: {affected}"
        )

    return affected, all_session_names


def compute_result(
    sessions: Sequence[str],
    kind: Literal["affected", "excluded"],
    output: str | None,
    *,
    verbose: bool = False,
) -> None:
    entries = globals()[f"make_{kind}_matrix_entries"](sessions, verbose=verbose)

    if output:
        with open(output, "w") as f:
            json.dump(entries, f, indent=2)
        rich.print(f"Saved {kind} test sessions entries of GitHub Actions matrix to '{output}'")
    else:
        rich.print(
            f"{kind} test sessions entries of GitHub Actions matrix: (use '--output <filename>' to save it)\n---------------------\n"
        )
        pretty.pprint(entries)


def make_affected_matrix_entries(
    affected_sessions: Sequence[str], *, verbose: bool = False
) -> list[dict[str, str]]:
    """Create a GitHub Actions matrix from the affected CPU-only sessions."""

    sessions = nox_sessions.get_sessions("-k", "cpu", verbose=verbose)
    entries: list[dict[str, str]] = []

    processed_sessions = set()
    for session in sessions:
        if session["name"] in affected_sessions:
            call_spec = tuple(session["call_spec"].items())
            session_id = (session["name"], *call_spec)

            if session_id not in processed_sessions:
                processed_sessions.add(session_id)
                session_args = ", ".join(session["call_spec"].values())
                if session_args:
                    session_args = f"({session_args})"
                entries.append(dict(name=session["name"], args=session_args))

    return entries


def make_excluded_matrix_entries(
    excluded_sessions: Sequence[str], *, verbose: bool = False
) -> list[dict[str, dict[str, str]]]:
    """Create a list of dicts with the exclusion pattern for a GitHub Actions matrix."""

    entries = [{"nox-session": {"name": session}} for session in excluded_sessions]
    return entries


@cli.command("matrix-exclude")
def matrix_exclude(
    *,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Sessions configuration file")
    ] = nox_sessions.DEFAULT_CONFIG,
    base_commit: Annotated[
        str | None, typer.Option("--base", "-b", help="Base commit for changes")
    ] = None,
    output: Annotated[str | None, typer.Option("--output", help="Output (JSON) file name.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Define excluded entries in a GitHub Actions matrix if not affected by changes from a base commit."""

    affected, all_sessions = collect_affected_sessions(
        config=config, base_commit=base_commit, verbose=verbose
    )
    excluded_sessions = sorted(set(all_sessions) - set(affected))
    compute_result(excluded_sessions, "excluded", output, verbose=verbose)


@cli.command("matrix-sessions")
def matrix_sessions(
    *,
    config: Annotated[
        str, typer.Option("--config", "-c", help="Sessions configuration file")
    ] = nox_sessions.DEFAULT_CONFIG,
    base_commit: Annotated[
        str | None, typer.Option("--base", "-b", help="Base commit for changes")
    ] = None,
    output: Annotated[str | None, typer.Option("--output", help="Output (JSON) file name.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Define entries of the test sessions affected by changes from a base commit as a factor definition in a GitHub Actions matrix."""

    affected, _all_sessions = collect_affected_sessions(
        config=config, base_commit=base_commit, verbose=verbose
    )
    compute_result(affected, "affected", output, verbose=verbose)


if __name__ == "__main__":
    cli()
