#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Update project dependencies and pre-commit hooks."""

from __future__ import annotations

import enum
import re
import subprocess

import packaging
import rich
import tomllib
import typer

from . import _common as common


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    UNRECOGNIZED_PYPROJECT_TOML = 10
    UNRECOGNIZED_PRECOMMIT_CONFIG = 11
    INVALID_NEW_VERSION_STRING = 12
    INVALID_CURRENT_VERSION_IN_ABOUT_PY = 13


cli = typer.Typer(no_args_is_help=True, name="update", help=__doc__)


@cli.command()
def dependencies() -> None:
    """Update project dependencies to their latest compatible versions."""
    subprocess.run("uv lock --upgrade", cwd=common.REPO_ROOT, shell=True, check=True)


@cli.command("pre-commit")
def precommit() -> None:
    """Update versions of pre-commit hooks."""
    subprocess.run(
        f"uv run --quiet --locked --project {common.REPO_ROOT} pre-commit autoupdate", shell=True
    )

    try:
        with open(common.REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            uv_spec = data["tool"]["uv"]["required-version"]
    except Exception as e:
        rich.print(
            f"Incompatible or missing 'tool.uv.required-version' key in 'pyproject.toml': {e}"
        )
        raise typer.Exit(ExitCode.UNRECOGNIZED_PYPROJECT_TOML) from e

    try:
        pre_commit_path = common.REPO_ROOT / ".pre-commit-config.yaml"
        with open(pre_commit_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = re.sub(
            r"additional_dependencies:\s*\[uv>=([\d\.]+)\]",
            f"additional_dependencies: [uv{uv_spec}]",
            content,
            count=1,
        )
        if new_content != content:
            rich.print(f"Updating required 'uv' version for uv-managed hooks to '{uv_spec}'")
            with open(pre_commit_path, "w", encoding="utf-8") as f:
                f.write(new_content)

    except Exception as e:
        rich.print(
            f"Incompatible or missing 'additional_dependencies' key for 'uv-managed-hook' "
            f"definitions in '.pre-commit-config.yaml': {e}"
        )
        raise typer.Exit(ExitCode.UNRECOGNIZED_PRECOMMIT_CONFIG) from e


@cli.command(name="package-version")
def package_version(new_version_number: str) -> None:
    """Hardcode a new version-in-git default version, for cases when git is not available."""
    try:
        packaging.version.Version(new_version_number)  # Validate version string
    except packaging.InvalidVersion as e:
        rich.print(f"[red]Error:[/red] '{new_version_number}' is not a valid version string.")
        raise typer.Exit(ExitCode.INVALID_NEW_VERSION_STRING) from e

    # Read the current default version from pyproject.toml
    try:
        with open(common.REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            current_version = data["tool"]["versioningit"]["default-version"]
    except Exception as e:
        rich.print(
            f"Incompatible or missing 'tool.versioningit.default-version' key in 'pyproject.toml': {e}"
        )
        raise typer.Exit(ExitCode.UNRECOGNIZED_PYPROJECT_TOML) from e

    new_version = f"{new_version_number!s}+unknown.version.details"

    # Hardcode the new default version in pyproject.toml
    pyproject_path = common.REPO_ROOT / "pyproject.toml"
    pyproject_path.write_text(
        pyproject_path.read_text().replace(f'"{current_version}"', f'"{new_version}"', 1)
    )

    # Hardcode the new default version in __about__.py
    about_path = common.REPO_ROOT / "src" / "gt4py" / "__about__.py"
    current_about_text = about_path.read_text()
    if f'"{current_version}"' not in current_about_text:
        rich.print(
            f"[red]Error:[/red] Current version '{current_version}' not found in '__about__.py'."
        )
        raise typer.Exit(ExitCode.INVALID_CURRENT_VERSION_IN_ABOUT_PY)

    about_path.write_text(
        current_about_text.replace(
            f'\non_build_version: Final = "{current_version}"',
            f'\non_build_version: Final = "{new_version}"',
            1,
        )
    )


@cli.command()
def all() -> None:  # noqa: A001 [builtin-variable-shadowing]
    """Update both project dependencies and pre-commit hooks."""
    dependencies()
    precommit()


if __name__ == "__main__":
    cli()
