#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Repository consistency checks."""

from __future__ import annotations

import ast
import collections
import typing

import rich
import typer
from helpers import common


cli = typer.Typer(no_args_is_help=True, name="check", help=__doc__)

SEARCH_ROOTS: typing.Final[tuple[str, ...]] = ("src", "tests", "scripts", "typing_tests")


def _typing_extensions_imports() -> dict[str, list[str]]:
    """Map every name imported from `typing_extensions` to where it is imported."""
    found: dict[str, list[str]] = collections.defaultdict(list)
    for root in SEARCH_ROOTS:
        for path in sorted((common.REPO_ROOT / root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing_extensions":
                    for alias in node.names:
                        location = f"{path.relative_to(common.REPO_ROOT)}:{node.lineno}"
                        found[alias.name].append(location)
    return dict(found)


@cli.command("typing-extensions-usage")
def typing_extensions_usage() -> None:
    """Report `typing_extensions` imports that the current Python floor makes redundant.

    The project imports from `typing_extensions` only what the supported Python floor
    does not provide, or where the two implementations genuinely differ. Both sets
    shrink as the floor rises, but nothing makes that visible: run this after bumping
    `requires-python` to find the imports that can move to `typing`.

    A name is only reported when `typing` provides the very same object, so the
    deliberate cases -- where `typing_extensions` is chosen because its implementation
    differs -- are not flagged.
    """
    try:
        import typing_extensions
    except ModuleNotFoundError:
        rich.print("[red]'typing_extensions' is not installed in this environment.[/red]")
        raise typer.Exit(1) from None

    floor = f"{common.PYTHON_VERSIONS[0]}"
    graduated: list[tuple[str, list[str]]] = []
    kept: list[str] = []
    for name, locations in sorted(_typing_extensions_imports().items()):
        in_typing = getattr(typing, name, None)
        if in_typing is None:
            kept.append(f"{name} (absent from 'typing')")
        elif in_typing is getattr(typing_extensions, name, None):
            graduated.append((name, locations))
        else:
            kept.append(f"{name} (different object in 'typing')")

    rich.print(f"Python floor: [bold]{floor}[/bold] (from '.python-versions')")
    if graduated:
        rich.print("\n[yellow]Can move to 'typing':[/yellow]")
        for name, locations in graduated:
            rich.print(f"  {name}")
            for location in locations:
                rich.print(f"      {location}")
    else:
        rich.print("\n[green]No 'typing_extensions' import can move to 'typing' yet.[/green]")
    if kept:
        rich.print("\nStaying on 'typing_extensions':")
        for entry in kept:
            rich.print(f"  {entry}")


if __name__ == "__main__":
    cli()
