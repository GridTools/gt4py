#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Migrate gt4py.next user code to dimension and connectivity classes (ADRs 0028, 0029).

Rewrites module-level declarations and the uses of Cartesian offsets:

    KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
    E2CDim = gtx.Dimension("E2C", gtx.DimensionKind.LOCAL)
    E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
    Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
    ... a(Koff[1]) ... as_offset(Koff, k_field) ...

becomes

    class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...
    class E2CDim(gtx.LocalDimensionIndex): ...
    class E2C(gtx.NeighborConnectivity[EdgeDim, CellDim]):
        Local = E2CDim
    ... a(KDim + 1) ... as_offset(KDim, k_field) ...

A connectivity adopts its existing local dimension (`Local = E2CDim`), so the names already used
for local dimensions keep working, and so do offsets that share a local dimension (`C2CE`
with `C2EDim`). What cannot be rewritten from the source alone is reported instead: offset
providers keyed by strings, which become keyed by the connectivity (`{E2C: table}`), `.value`
on a dimension (now `.tag`), and `isinstance` checks against `Dimension`.

The output is not formatted; run `ruff format` on the changed files afterwards.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import pathlib
import re
from collections.abc import Iterable, Iterator

import typer


cli = typer.Typer(no_args_is_help=True, name="migrate-connectivities", help=__doc__)


@dataclasses.dataclass(frozen=True)
class Edit:
    """Replace source lines `[start, end)` (0-based) with `text`."""

    start: int
    end: int
    text: str


@dataclasses.dataclass
class Module:
    path: pathlib.Path
    source: str
    tree: ast.Module
    edits: list[Edit] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    #: Names of `gt4py.next` the migrated declarations use unqualified, to be imported.
    needed: set[str] = dataclasses.field(default_factory=set)

    @property
    def lines(self) -> list[str]:
        return self.source.splitlines(keepends=True)

    def segment(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        assert segment is not None
        return segment

    def note(self, node: ast.stmt | ast.expr, message: str) -> None:
        self.notes.append(f"{self.path}:{node.lineno}: {message}")


def _callee_name(call: ast.Call) -> tuple[str, str] | None:
    """`(prefix, name)` of a call to `[prefix.]name`, e.g. `("gtx.", "Dimension")`."""
    match call.func:
        case ast.Name(id=name):
            return "", name
        case ast.Attribute(value=value, attr=name):
            return f"{ast.unparse(value)}.", name
    return None


def _keyword(call: ast.Call, name: str, position: int) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return call.args[position] if len(call.args) > position else None


_DECLARATION_CALLEES = ("Dimension", "FieldOffset")


def _imported_aliases(module: Module) -> dict[str, str]:
    """Local names of imported `Dimension` / `FieldOffset`, e.g. `{"FO": "FieldOffset"}`."""
    return {
        alias.asname or alias.name: alias.name
        for statement in ast.walk(module.tree)
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if alias.name in _DECLARATION_CALLEES
    }


def _declarations(module: Module) -> Iterator[tuple[ast.Assign, str, ast.Call, str, str]]:
    """Module-level `name = [prefix.]Dimension(...)` / `FieldOffset(...)` statements."""
    aliases = _imported_aliases(module)
    for statement in module.tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and (callee := _callee_name(statement.value)) is not None
        ):
            prefix, name = callee
            name = aliases.get(name, name) if prefix == "" else name
            if name in _DECLARATION_CALLEES:
                yield statement, statement.targets[0].id, statement.value, prefix, name


def _replace(module: Module, statement: ast.stmt, text: str) -> None:
    assert statement.end_lineno is not None
    module.edits.append(Edit(statement.lineno - 1, statement.end_lineno, text))


def _migrate_declarations(module: Module, cartesian: dict[str, str]) -> None:
    for statement, name, call, prefix, kind_of_call in _declarations(module):
        if kind_of_call == "Dimension":
            kind = _keyword(call, "kind", 1)
            kind_src = module.segment(kind) if kind is not None else None
            if kind_src is not None and kind_src.split(".")[-1] == "LOCAL":
                base = "LocalDimensionIndex"
                text = f"class {name}({prefix}{base}): ...\n"
            elif kind_src is None:
                base = "DimensionIndex"
                text = f"class {name}({prefix}{base}): ...\n"
            else:
                base = "DimensionIndex"
                text = f"class {name}({prefix}{base}, kind={kind_src}): ...\n"
            if not prefix:
                module.needed.add(base)
            _replace(module, statement, text)
            continue

        source, target = _keyword(call, "source", 1), _keyword(call, "target", 2)
        if source is None or not isinstance(target, ast.Tuple):
            module.note(statement, f"'{name}': unrecognized 'FieldOffset' arguments, not migrated.")
            continue
        if len(target.elts) == 2:
            origin, local = (module.segment(element) for element in target.elts)
            text = (
                f"class {name}({prefix}NeighborConnectivity[{origin}, {module.segment(source)}]):\n"
                f"    Local = {local}\n"
            )
            if not prefix:
                module.needed.add("NeighborConnectivity")
            _replace(module, statement, text)
        elif len(target.elts) == 1 and module.segment(target.elts[0]) == module.segment(source):
            # A Cartesian offset has no declaration any more: `Off[i]` is `Dim + i`.
            cartesian[name] = module.segment(source)
            _replace(module, statement, "")
        else:
            module.note(statement, f"'{name}': a cross-dimension offset has no class equivalent.")


def _migrate_imports(module: Module) -> None:
    """Drop imports of the removed `FieldOffset`; import the class names used unqualified."""
    fieldoffset_names = {
        local for local, name in _imported_aliases(module).items() if name == "FieldOffset"
    }
    imported = {
        alias.asname or alias.name
        for statement in ast.walk(module.tree)
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    }
    missing = sorted(module.needed - imported)
    added = False
    for statement in module.tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        names = {alias.asname or alias.name for alias in statement.names}
        drops = names & fieldoffset_names
        adds_here = bool(missing) and not added and bool(names & {"Dimension", "DimensionKind"})
        if not (drops or adds_here):
            continue
        kept = [
            ast.unparse(alias)
            for alias in statement.names
            if (alias.asname or alias.name) not in fieldoffset_names
        ]
        text = (
            f"from {'.' * statement.level}{statement.module or ''} import {', '.join(kept)}\n"
            if kept
            else ""
        )
        if adds_here:
            text += f"from gt4py.next import {', '.join(missing)}\n"
            added = True
        _replace(module, statement, text)
    if missing and not added:
        module.notes.append(
            f"{module.path}: import {', '.join(missing)} from 'gt4py.next', used by the migrated"
            " declarations."
        )


class _CartesianUses(ast.NodeVisitor):
    """Rewrite the uses of removed Cartesian offsets, and note the ones it cannot."""

    def __init__(self, module: Module, cartesian: dict[str, str]) -> None:
        self.module = module
        self.cartesian = cartesian
        #: (line, start column, end column, replacement) of single-line expression rewrites
        self.rewrites: list[tuple[int, int, int, str]] = []
        self.handled: set[int] = set()
        #: names bound in the enclosing function scopes, which shadow a removed offset
        self.shadowed: list[set[str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        arguments = node.args
        bound = {
            argument.arg
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        } | {
            name.id
            for name in ast.walk(node)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        }
        self.shadowed.append(bound)
        self.generic_visit(node)
        self.shadowed.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def _rewrite(self, node: ast.expr, text: str) -> None:
        assert node.end_lineno is not None and node.end_col_offset is not None
        if node.lineno != node.end_lineno:
            self.module.note(node, f"multi-line expression; rewrite by hand as '{text}'.")
            return
        self.rewrites.append((node.lineno - 1, node.col_offset, node.end_col_offset, text))

    def _dimension_of(self, node: ast.expr) -> str | None:
        """The dimension replacing `Off` or `module.Off`, qualified like the offset was."""
        match node:
            case ast.Name(id=name) if name in self.cartesian and not any(
                name in bound for bound in self.shadowed
            ):
                return self.cartesian[name]
            case ast.Attribute(value=value, attr=name) if name in self.cartesian:
                return f"{self.module.segment(value)}.{self.cartesian[name]}"
        return None

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (dim := self._dimension_of(node.value)) is not None:
            match node.slice:
                case ast.Constant(value=int() as index):
                    text = f"{dim} + {index}" if index >= 0 else f"{dim} - {-index}"
                case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() as index)):
                    text = f"{dim} - {index}"
                case _:
                    text = f"{dim} + ({self.module.segment(node.slice)})"
            self._rewrite(node, text)
            self.handled.add(id(node.value))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if id(node) not in self.handled and (dim := self._dimension_of(node)) is not None:
            # e.g. `as_offset(Koff, field)`, which now takes the dimension
            self._rewrite(node, dim)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) not in self.handled and (dim := self._dimension_of(node)) is not None:
            self._rewrite(node, dim)
        else:
            self.generic_visit(node)


def _migrate_cartesian_uses(module: Module, cartesian: dict[str, str]) -> None:
    if not cartesian:
        return
    visitor = _CartesianUses(module, cartesian)
    import_lines: set[int] = set()
    for statement in ast.walk(module.tree):
        if isinstance(statement, ast.ImportFrom) and any(
            alias.name in cartesian for alias in statement.names
        ):
            import_lines.update(range(statement.lineno - 1, statement.end_lineno or 0))
            names = [
                cartesian.get(alias.name, alias.name) if alias.asname is None else alias.name
                for alias in statement.names
            ]
            unique = list(dict.fromkeys(names))
            _replace(
                module,
                statement,
                " " * statement.col_offset
                + f"from {'.' * statement.level}{statement.module or ''} import {', '.join(unique)}\n",
            )
    for statement in module.tree.body:
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            visitor.visit(statement)
        match statement:
            case ast.Assign(targets=[ast.Name(id="__all__")], value=ast.List(elts=all_names)):
                for entry in all_names:
                    if isinstance(entry, ast.Constant) and entry.value in cartesian:
                        module.note(entry, f"'__all__' lists the removed offset '{entry.value}'.")

    lines = module.lines
    for line, start, end, text in sorted(visitor.rewrites, reverse=True):
        if line in import_lines:
            continue
        lines[line] = lines[line][:start] + text + lines[line][end:]
    module.source = "".join(lines)


_PROVIDER_KEY_RE = re.compile(r"""(?P<quote>["'])(?P<name>[A-Za-z_]\w*)(?P=quote)\s*:""")


def _report(
    module: Module,
    offset_keys: dict[str, str],
    cartesian: dict[str, str],
    dimension_names: set[str],
) -> None:
    for number, line in enumerate(module.source.splitlines(), start=1):
        for match in _PROVIDER_KEY_RE.finditer(line):
            if (connectivity := offset_keys.get(match["name"])) is None:
                continue
            if connectivity in cartesian:
                message = (
                    f"offset-provider key '{match['name']}': remove the entry, a Cartesian shift"
                    f" ('{cartesian[connectivity]} + i') needs none."
                )
            else:
                message = (
                    f"offset-provider key '{match['name']}' is keyed by the connectivity class"
                    f" now, e.g. '{{{connectivity}: table}}'."
                )
            module.notes.append(f"{module.path}:{number}: {message}")
    for node in ast.walk(module.tree):
        match node:
            case ast.Attribute(
                value=ast.Name(id=name) | ast.Attribute(attr=name), attr="value"
            ) if name in dimension_names:
                module.note(node, f"'{name}.value': a dimension's name is '{name}.tag' now.")
            case ast.Call(
                func=ast.Name(id="isinstance"), args=[_, ast.Attribute(attr="Dimension")]
            ):
                module.note(
                    node,
                    "'isinstance(..., Dimension)': a dimension is a class now; use"
                    " 'isinstance(obj, gt4py.next.common.DimensionMeta)'.",
                )


def _apply(module: Module) -> str:
    lines = module.lines
    for edit in sorted(module.edits, key=lambda edit: edit.start, reverse=True):
        lines[edit.start : edit.end] = [edit.text] if edit.text else []
    return "".join(lines)


def migrate(sources: dict[pathlib.Path, str]) -> tuple[dict[pathlib.Path, str], list[str]]:
    """Migrate the given modules; return their new sources and what is left to do by hand."""
    modules = [
        Module(path=path, source=source, tree=ast.parse(source)) for path, source in sources.items()
    ]
    cartesian: dict[str, str] = {}
    #: offset-provider keys that name a `FieldOffset`: its variable name and its tag, if different
    offset_keys: dict[str, str] = {}
    dimension_names: set[str] = set()
    for module in modules:
        for _, name, call, _, kind_of_call in _declarations(module):
            if kind_of_call == "Dimension":
                dimension_names.add(name)
                continue
            offset_keys[name] = name
            if call.args and isinstance(tag := call.args[0], ast.Constant):
                offset_keys[str(tag.value)] = name
        _migrate_declarations(module, cartesian)
        _migrate_imports(module)

    results: dict[pathlib.Path, str] = {}
    notes: list[str] = []
    for module in modules:
        _report(module, offset_keys, cartesian, dimension_names)
        migrated = _apply(module)
        # Cartesian uses are rewritten on the migrated text, re-parsed, so that line numbers
        # refer to what the declaration edits left.
        second = Module(path=module.path, source=migrated, tree=ast.parse(migrated))
        _migrate_cartesian_uses(second, cartesian)
        results[module.path] = _apply(second)
        notes += module.notes + second.notes
    return results, notes


def _python_files(paths: Iterable[pathlib.Path]) -> Iterator[pathlib.Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


@cli.command()
def run(
    paths: list[pathlib.Path],
    write: bool = typer.Option(False, "--write", help="Rewrite files instead of printing a diff."),
) -> None:
    """Migrate the Python files in PATHS (directories are searched recursively)."""
    sources = {path: path.read_text() for path in _python_files(paths)}
    results, notes = migrate(sources)
    for path, new in results.items():
        if new == sources[path]:
            continue
        if write:
            path.write_text(new)
            typer.echo(f"migrated {path}")
        else:
            typer.echo(
                "".join(
                    difflib.unified_diff(
                        sources[path].splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=str(path),
                        tofile=str(path),
                    )
                )
            )
    if notes:
        typer.echo("\nLeft to migrate by hand (line numbers of the original files):", err=True)
        for note in notes:
            typer.echo(f"  {note}", err=True)


if __name__ == "__main__":
    cli()
