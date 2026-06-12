#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Raise the lower or upper supported Python version across the project.

The supported versions are enumerated in the '.python-versions' file (the
single source of truth from which the noxfile and the GitHub Actions workflows
derive their version matrices). This script keeps that file in sync with all
the places that hardcode the supported range: project metadata and dev-tool
configuration in 'pyproject.toml', the 'nox.options.sessions' list in
'noxfile.py', the CSCS CI bounds in 'ci/cscs-ci.yml', and the documentation.

Use '--dry-run' to preview every change without writing files or running tools.
"""

from __future__ import annotations

import difflib
import enum
import pathlib
import re
import subprocess
from typing import Annotated, Callable

import packaging.version
import rich
import rich.markup
import tomllib
import typer
from helpers import common


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    UNRECOGNIZED_PYPROJECT_TOML = 10
    INVALID_VERSION_STRING = 12
    VERSION_ALREADY_SUPPORTED = 20
    CANNOT_DROP_LAST_VERSION = 21
    NEW_VERSION_NOT_HIGHEST = 22
    UNRECOGNIZED_NOXFILE = 30
    TOOL_RUN_FAILED = 40


cli = typer.Typer(no_args_is_help=True, name="python-support", help=__doc__)


# -- Version helpers --
def _parts(version: str) -> tuple[int, int]:
    """Return the '(major, minor)' tuple of a 'X.Y' version string."""
    try:
        release = packaging.version.Version(version).release
    except packaging.version.InvalidVersion as e:
        rich.print(f"[red]Error:[/red] '{version}' is not a valid version string.")
        raise typer.Exit(ExitCode.INVALID_VERSION_STRING) from e
    return (release[0], release[1] if len(release) > 1 else 0)


def _nodot(version: str) -> str:
    """Turn '3.10' into '310' (used for ruff's 'target-version')."""
    major, minor = _parts(version)
    return f"{major}{minor}"


def _next_minor(version: str) -> str:
    """Return the next minor version, e.g. '3.15' -> '3.16'."""
    major, minor = _parts(version)
    return f"{major}.{minor + 1}"


# -- File-edit accumulator --
class Edits:
    """Collect in-memory edits so they can be previewed or written atomically."""

    def __init__(self) -> None:
        self._originals: dict[pathlib.Path, str] = {}
        self._current: dict[pathlib.Path, str] = {}

    def _load(self, path: pathlib.Path) -> str:
        if path not in self._current:
            text = path.read_text(encoding="utf-8")
            self._originals[path] = text
            self._current[path] = text
        return self._current[path]

    def get(self, path: pathlib.Path) -> str:
        return self._load(path)

    def edit(self, path: pathlib.Path, fn: Callable[[str], str]) -> None:
        """Apply 'fn' to the current text of 'path'."""
        self._current[path] = fn(self._load(path))

    def sub(self, path: pathlib.Path, pattern: str, repl: str, *, count: int = 1) -> None:
        """Apply a single regex substitution, warning if nothing matched."""

        def _apply(text: str) -> str:
            new_text, n = re.subn(pattern, repl, text, count=count)
            if n == 0:
                rich.print(
                    f"[yellow]Warning:[/yellow] pattern not found in "
                    f"'{path.relative_to(common.REPO_ROOT)}': {pattern!r} (skipped)"
                )
            return new_text

        self.edit(path, _apply)

    @property
    def changed(self) -> dict[pathlib.Path, tuple[str, str]]:
        return {
            path: (self._originals[path], current)
            for path, current in self._current.items()
            if current != self._originals[path]
        }

    def preview(self) -> None:
        for path, (old, new) in self.changed.items():
            rel = path.relative_to(common.REPO_ROOT)
            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            rich.print(f"[bold]{rel}[/bold]")
            for line in diff:
                content = rich.markup.escape(line.rstrip("\n"))
                if line.startswith("+") and not line.startswith("+++"):
                    rich.print(f"[green]{content}[/green]")
                elif line.startswith("-") and not line.startswith("---"):
                    rich.print(f"[red]{content}[/red]")
                else:
                    rich.print(content)
            rich.print()

    def write(self) -> None:
        for path, (_old, new) in self.changed.items():
            path.write_text(new, encoding="utf-8")
            rich.print(f"  updated [bold]{path.relative_to(common.REPO_ROOT)}[/bold]")


# -- Individual file editors --
def _edit_python_versions(edits: Edits, *, drop: str | None, add: str | None) -> None:
    path = common.REPO_ROOT / ".python-versions"

    def _apply(text: str) -> str:
        lines = text.splitlines()
        if drop is not None:
            lines = [line for line in lines if line.strip() != drop]
        if add is not None:
            last_version_idx = max(
                i for i, line in enumerate(lines) if (v := line.strip()) and not v.startswith("#")
            )
            lines.insert(last_version_idx + 1, add)
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    edits.edit(path, _apply)


def _edit_pyproject_lower(edits: Edits, *, dropped: str, new_min: str) -> None:
    path = common.REPO_ROOT / "pyproject.toml"
    # Remove the classifier for the dropped version.
    edits.sub(
        path,
        rf"(?m)^[ \t]*'Programming Language :: Python :: {re.escape(dropped)}',\n",
        "",
    )
    # Bump only the '>=' lower bound, preserving '<...' and '!=...' exclusions.
    _bump_requires_python(edits, path, rf">={re.escape(dropped)}", f">={new_min}")
    # ruff target-version tracks the floor.
    edits.sub(path, r"target-version = 'py\d+'", f"target-version = 'py{_nodot(new_min)}'")
    # mypy should target the lowest supported version.
    _set_mypy_python_version(edits, path, new_min)


def _edit_pyproject_upper(edits: Edits, *, current_max: str, new_version: str) -> None:
    path = common.REPO_ROOT / "pyproject.toml"
    # Add the classifier for the new highest version after the current highest.
    edits.sub(
        path,
        rf"( *'Programming Language :: Python :: {re.escape(current_max)}',\n)",
        rf"\1  'Programming Language :: Python :: {new_version}',\n",
    )
    # Bump only the '<' upper bound to the next minor after the new version.
    _bump_requires_python(edits, path, r"<\d+\.\d+", f"<{_next_minor(new_version)}")


def _bump_requires_python(edits: Edits, path: pathlib.Path, pattern: str, repl: str) -> None:
    """Substitute a single bound token inside the 'requires-python' value only."""

    def _apply(text: str) -> str:
        m = re.search(r"(requires-python = ')([^']*)(')", text)
        if not m:
            rich.print("[yellow]Warning:[/yellow] 'requires-python' not found (skipped)")
            return text
        new_value, n = re.subn(pattern, repl, m.group(2), count=1)
        if n == 0:
            rich.print(
                f"[yellow]Warning:[/yellow] bound {pattern!r} not found in 'requires-python' "
                f"(skipped)"
            )
            return text
        return text[: m.start()] + m.group(1) + new_value + m.group(3) + text[m.end() :]

    edits.edit(path, _apply)


def _set_mypy_python_version(edits: Edits, path: pathlib.Path, version: str) -> None:
    def _apply(text: str) -> str:
        # Replace an existing 'python_version' inside '[tool.mypy]' if present.
        block = re.search(r"\[tool\.mypy\][^\[]*", text)
        if block and re.search(r"^python_version = ", block.group(0), re.MULTILINE):
            start = block.start()
            updated = re.sub(
                r"^python_version = .*$",
                f"python_version = '{version}'",
                block.group(0),
                count=1,
                flags=re.MULTILINE,
            )
            return text[:start] + updated + text[block.end() :]
        return re.sub(
            r"(\[tool\.mypy\]\n)",
            rf"\1python_version = '{version}'\n",
            text,
            count=1,
        )

    edits.edit(path, _apply)


def _enable_ruff_pyupgrade(edits: Edits) -> None:
    """Add the 'UP' (pyupgrade) ruleset to ruff's 'select' list if absent."""
    path = common.REPO_ROOT / "pyproject.toml"

    def _apply(text: str) -> str:
        m = re.search(r"^select = \[([^\]]*)\]", text, re.MULTILINE)
        if not m:
            rich.print("[yellow]Warning:[/yellow] ruff 'select' list not found (skipped)")
            return text
        if re.search(r"'UP'", m.group(1)):
            return text
        rich.print("Enabling ruff 'UP' (pyupgrade) ruleset")
        updated = m.group(0).replace("'YTT'", "'UP', 'YTT'", 1)
        return text[: m.start()] + updated + text[m.end() :]

    edits.edit(path, _apply)


def _rebuild_nox_sessions(edits: Edits, target_versions: list[str]) -> None:
    path = common.REPO_ROOT / "noxfile.py"

    def _apply(text: str) -> str:
        m = re.search(r"(nox\.options\.sessions = \[\n)(.*?)(\n\])", text, re.DOTALL)
        if not m:
            rich.print("[red]Error:[/red] 'nox.options.sessions' list not found.")
            raise typer.Exit(ExitCode.UNRECOGNIZED_NOXFILE)

        entry_re = re.compile(r'^\s*"(?P<entry>[^"]+)",?\s*$')
        item_re = re.compile(r"^(?P<family>[a-z0-9_]+)-(?P<ver>\d+\.\d+)(?P<suffix>.*)$")
        families: list[str] = []
        templates: dict[str, dict[str, list[str]]] = {}
        for line in m.group(2).splitlines():
            entry_m = entry_re.match(line)
            if not entry_m:
                continue
            item_m = item_re.match(entry_m.group("entry"))
            if not item_m:
                continue
            family, ver, suffix = item_m.group("family", "ver", "suffix")
            if family not in templates:
                families.append(family)
                templates[family] = {}
            templates[family].setdefault(ver, []).append(suffix)

        out_lines: list[str] = []
        for family in families:
            present = templates[family]
            max_ver = max(present, key=_parts)
            for ver in sorted(target_versions, key=_parts):
                for suffix in present[max_ver]:
                    out_lines.append(f'    "{family}-{ver}{suffix}",')
        return text[: m.start()] + m.group(1) + "\n".join(out_lines) + m.group(3) + text[m.end() :]

    edits.edit(path, _apply)


def _edit_cscs_ci(edits: Edits, new_min: str, new_max: str) -> None:
    path = common.REPO_ROOT / "ci" / "cscs-ci.yml"
    edits.sub(
        path,
        r"(&test_python_versions )\['[\d.]+', '[\d.]+'\]",
        rf"\1['{new_min}', '{new_max}']",
    )


def _edit_docs(edits: Edits, new_min: str, new_max: str, *, floor_changed: bool) -> None:
    # The en dash (U+2013) is intentional: it must match the AGENTS.md text verbatim.
    edits.sub(
        common.REPO_ROOT / "AGENTS.md",
        r"\*\*Python [\d.]+–[\d.]+\*\*",  # noqa: RUF001 [ambiguous-unicode-character-string]
        f"**Python {new_min}–{new_max}**",  # noqa: RUF001 [ambiguous-unicode-character-string]
    )
    if floor_changed:
        edits.sub(
            common.REPO_ROOT / "CONTRIBUTING.md",
            r"test_cartesian-[\d.]+\(internal, cpu\)",
            f"test_cartesian-{new_min}(internal, cpu)",
        )


def _add_changelog_entry(edits: Edits, message: str) -> None:
    path = common.REPO_ROOT / "CHANGELOG.md"

    def _apply(text: str) -> str:
        bullet = f"- {message}"
        if bullet in text:
            return text
        unreleased = re.search(r"## \[Unreleased\].*?(?=\n## \[|\Z)", text, re.DOTALL)
        if unreleased:
            section = unreleased.group(0)
            if "### General" in section:
                new_section = re.sub(r"(### General\n\n)", rf"\1{bullet}\n", section, count=1)
            else:
                new_section = section.rstrip() + f"\n\n### General\n\n{bullet}\n"
            return text[: unreleased.start()] + new_section + text[unreleased.end() :]
        block = f"## [Unreleased]\n\n### General\n\n{bullet}\n\n"
        return re.sub(r"(\n## \[)", rf"\n{block}## [", text, count=1)

    edits.edit(path, _apply)


def _warn_dependency_markers(boundary: str) -> None:
    """Report 'python_version' env markers that may need manual review after a bump."""
    data = tomllib.loads((common.REPO_ROOT / "pyproject.toml").read_text())
    project = data.get("project", {})
    groups: dict[str, list[str]] = {"dependencies": project.get("dependencies", [])}
    for name, reqs in project.get("optional-dependencies", {}).items():
        groups[f"optional-dependencies.{name}"] = reqs

    marker_re = re.compile(r"python_version\s*(<|>=|<=|>|==)\s*[\"']([\d.]+)[\"']")
    findings: list[str] = []
    for group, reqs in groups.items():
        for req in reqs:
            if marker_re.search(req):
                findings.append(f"  [{group}] {req}")

    if findings:
        rich.print(
            f"\n[yellow]Review dependency markers[/yellow] (boundary near {boundary}; "
            f"not edited automatically):"
        )
        for finding in findings:
            rich.print(finding)


def _validate_pyproject(edits: Edits) -> None:
    path = common.REPO_ROOT / "pyproject.toml"
    try:
        tomllib.loads(edits.get(path))
    except tomllib.TOMLDecodeError as e:
        rich.print(f"[red]Error:[/red] edits produced invalid 'pyproject.toml': {e}")
        raise typer.Exit(ExitCode.UNRECOGNIZED_PYPROJECT_TOML) from e


def _run_tools(*, modernize: bool) -> None:
    rich.print("\n[bold]Regenerating lockfile[/bold] (uv lock)...")
    try:
        subprocess.run("uv lock", cwd=common.REPO_ROOT, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        rich.print(f"[red]Error:[/red] 'uv lock' failed: {e}")
        raise typer.Exit(ExitCode.TOOL_RUN_FAILED) from e

    if modernize:
        rich.print("\n[bold]Modernizing codebase[/bold] (ruff check --fix, ruff format)...")
        # ruff returns non-zero when it applies fixes, so do not use 'check=True'.
        subprocess.run(
            "uv run --group dev --frozen ruff check --fix", cwd=common.REPO_ROOT, shell=True
        )
        subprocess.run("uv run --group dev --frozen ruff format", cwd=common.REPO_ROOT, shell=True)
    else:
        rich.print(
            "\n[dim]Skipped code modernization (pass '--modernize' to enable the 'UP' "
            "ruleset and auto-refactor to the new language floor).[/dim]"
        )


def _finalize(edits: Edits, *, dry_run: bool, yes: bool, run_tools: bool, modernize: bool) -> None:
    """Common preview/confirm/write/run pipeline for both commands."""
    _validate_pyproject(edits)

    if not edits.changed:
        rich.print("[yellow]Nothing to change.[/yellow]")
        return

    if dry_run:
        rich.print("[bold]Planned changes (dry-run, nothing written):[/bold]\n")
        edits.preview()
        return

    rich.print("[bold]Files to update:[/bold]")
    for path in edits.changed:
        rich.print(f"  {path.relative_to(common.REPO_ROOT)}")
    if not yes:
        typer.confirm("\nApply these changes?", abort=True)

    edits.write()

    if run_tools:
        _run_tools(modernize=modernize)


# -- Commands --
@cli.command("bump-lower")
def bump_lower(
    *,
    modernize: Annotated[
        bool,
        typer.Option(
            "--modernize", help="Enable ruff 'UP' rules and auto-refactor to the new floor"
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without writing or running tools")
    ] = False,
    run_tools: Annotated[
        bool, typer.Option("--no-run-tools/--run-tools", help="Skip 'uv lock' and tool runs")
    ] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")] = False,
) -> None:
    """Raise the lower bound by dropping the current lowest supported Python version."""
    versions = common.PYTHON_VERSIONS
    if len(versions) <= 1:
        rich.print("[red]Error:[/red] cannot drop the only remaining Python version.")
        raise typer.Exit(ExitCode.CANNOT_DROP_LAST_VERSION)

    dropped, new_min, new_max = versions[0], versions[1], versions[-1]
    rich.print(
        f"Raising lower bound: dropping Python [red]{dropped}[/red], "
        f"new supported range [green]{new_min}-{new_max}[/green]"
    )

    edits = Edits()
    _edit_python_versions(edits, drop=dropped, add=None)
    _edit_pyproject_lower(edits, dropped=dropped, new_min=new_min)
    if modernize:
        _enable_ruff_pyupgrade(edits)
    _rebuild_nox_sessions(edits, versions[1:])
    _edit_cscs_ci(edits, new_min, new_max)
    _edit_docs(edits, new_min, new_max, floor_changed=True)
    _add_changelog_entry(edits, f"Drop support for Python {dropped}.")
    _warn_dependency_markers(new_min)

    _finalize(edits, dry_run=dry_run, yes=yes, run_tools=run_tools, modernize=modernize)


@cli.command("bump-upper")
def bump_upper(
    new_version: Annotated[str, typer.Argument(help="New highest supported version, e.g. '3.15'")],
    *,
    modernize: Annotated[
        bool,
        typer.Option(
            "--modernize", help="Enable ruff 'UP' rules and auto-refactor (floor unchanged)"
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without writing or running tools")
    ] = False,
    run_tools: Annotated[
        bool, typer.Option("--no-run-tools/--run-tools", help="Skip 'uv lock' and tool runs")
    ] = True,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")] = False,
) -> None:
    """Raise the upper bound by adding a new highest supported Python version."""
    versions = common.PYTHON_VERSIONS
    if new_version in versions:
        rich.print(f"[red]Error:[/red] Python {new_version} is already supported.")
        raise typer.Exit(ExitCode.VERSION_ALREADY_SUPPORTED)
    if _parts(new_version) <= _parts(versions[-1]):
        rich.print(
            f"[red]Error:[/red] {new_version} is not greater than the current highest "
            f"version ({versions[-1]}); only adding a new highest version is supported."
        )
        raise typer.Exit(ExitCode.NEW_VERSION_NOT_HIGHEST)

    new_min, current_max = versions[0], versions[-1]
    rich.print(
        f"Raising upper bound: adding Python [green]{new_version}[/green], "
        f"new supported range [green]{new_min}-{new_version}[/green]"
    )

    edits = Edits()
    _edit_python_versions(edits, drop=None, add=new_version)
    _edit_pyproject_upper(edits, current_max=current_max, new_version=new_version)
    if modernize:
        _enable_ruff_pyupgrade(edits)
    _rebuild_nox_sessions(edits, [*versions, new_version])
    _edit_cscs_ci(edits, new_min, new_version)
    _edit_docs(edits, new_min, new_version, floor_changed=False)
    _add_changelog_entry(edits, f"Add support for Python {new_version}.")
    _warn_dependency_markers(new_version)

    _finalize(edits, dry_run=dry_run, yes=yes, run_tools=run_tools, modernize=modernize)


if __name__ == "__main__":
    cli()
