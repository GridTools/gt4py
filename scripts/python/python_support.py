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
@cli.command("raise-lower")
def raise_lower(
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


@cli.command("raise-upper")
def raise_upper(
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