#! /usr/bin/env -S uv run -q --script
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#


"""Script for generating test session names based on git changes."""

from __future__ import annotations

import json
import pathlib
from fnmatch import fnmatch
from typing import Any, Final

import git
from git.exc import GitCommandError
import typer
import yaml

ROOT_DIR: Final = pathlib.Path(__file__).parent

app = typer.Typer(no_args_is_help=True)


def _load_noxfile_config(config_path: pathlib.Path) -> dict[str, Any]:
    """Load and parse the noxfile-ci.yaml configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_changed_files(repo_path: pathlib.Path, base_commit: str) -> list[str]:
    """Get list of changed files between base_commit and current HEAD."""
    repo = git.Repo(repo_path)
    
    # Get the diff between base_commit and HEAD
    diff = repo.git.diff("--name-only", f"{base_commit}..HEAD")
    
    if not diff.strip():
        return []
    
    return diff.strip().split('\n')


def _matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if a file path matches a glob pattern."""
    # Handle both Unix-style paths and patterns
    return fnmatch(file_path, pattern) or fnmatch(file_path.replace('\\', '/'), pattern)


def _should_run_session(changed_files: list[str], session: dict[str, Any]) -> bool:
    """Determine if a session should run based on changed files and session patterns."""
    
    # If session has 'paths' field, at least one changed file must match
    if "paths" in session:
        patterns = session["paths"]
        for file_path in changed_files:
            for pattern in patterns:
                if _matches_pattern(file_path, pattern):
                    return True
        return False
    
    # If session has 'paths-ignore' field, no changed files should match
    if "paths-ignore" in session:
        patterns = session["paths-ignore"]
        for file_path in changed_files:
            for pattern in patterns:
                if _matches_pattern(file_path, pattern):
                    return False
        return True
    
    # If no path patterns are specified, always run
    return True


@app.command()
def list_sessions(
    base_commit: str = typer.Argument(..., help="Base commit to compare changes against"),
    output: str = typer.Option("sessions.json", "--output", "-o", help="Output JSON file path"),
    config: str = typer.Option("noxfile-ci.yaml", "--config", "-c", help="Path to noxfile-ci.yaml config"),
) -> None:
    """Generate list of test session names based on git changes since base_commit."""
    
    config_path = ROOT_DIR / config
    repo_path = ROOT_DIR
    output_path = pathlib.Path(output)
    
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config_data = _load_noxfile_config(config_path)
        
        # Get changed files
        changed_files = _get_changed_files(repo_path, base_commit)
        
        # Determine which sessions should run
        sessions_to_run = []
        
        for session in config_data.get("sessions", []):
            if _should_run_session(changed_files, session):
                sessions_to_run.append(session["name"])
        
        # Output results as JSON
        result = {
            "base_commit": base_commit,
            "changed_files": changed_files,
            "sessions": sessions_to_run
        }
        
        # Write to output file
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        typer.echo(f"Generated session list: {output_path}")
        typer.echo(f"Sessions to run: {', '.join(sessions_to_run) if sessions_to_run else 'none'}")
        
    except GitCommandError as e:
        typer.echo(f"Git error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()