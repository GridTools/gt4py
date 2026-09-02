# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Executable check of the layering inside `gt4py.next.otf`.

The DSL-agnostic half of the toolchain (artifact models, build systems,
compilers, binding interface) must not pull in the GT4Py IRs. That boundary
cannot be observed by importing the modules: every one of them does
``from gt4py.next import config``, which executes the `gt4py.next` package
`__init__` and therefore loads the whole frontend regardless of what the
module itself declares. So the graph is reconstructed from the AST instead,
counting only the imports that a real interpreter would execute when the
module is loaded. Two kinds of import are therefore not edges:

- `if TYPE_CHECKING:` blocks, which never execute.
- imports inside a function body, which execute only if that function is
  called. Deferring an import into a function is the sanctioned way to use a
  higher layer from a lower one without taking an import-time dependency on
  it, so counting those would forbid the very escape hatch this boundary
  relies on (`instrumentation.stage_dump` formats FOAST nodes that way).

Class bodies and module-level conditionals *are* entered: both execute while
the module is being loaded.
"""

from __future__ import annotations

import ast
import collections
import pathlib
from collections.abc import Iterable, Iterator, Mapping, Set

import pytest

import gt4py


SRC_ROOT = pathlib.Path(gt4py.__file__).resolve().parent.parent

#: Packages that hold the GT4Py IRs and the DSL frontend.
IR_PACKAGES = ("gt4py.next.iterator", "gt4py.next.ffront")

#: Modules that must stay clear of the IRs. `otf.compilation` is expanded to
#: all its submodules so that a newly added file is covered automatically.
IR_FREE_MODULES = (
    "gt4py.next.otf.workflow",
    "gt4py.next.otf.artifacts",
    "gt4py.next.otf.binding.interface",
    "gt4py.next.otf.runners",
    "gt4py.next.otf.compilation.*",
)

#: `binding.cpp_interface` uses `type_system.type_info.is_compatible_type`,
#: which still special-cases iterator types. See the TODO(tehrengruber) on
#: that function in `src/gt4py/next/type_system/type_info.py` (line 429):
#: until it is split, `otf.binding` cannot be IR-free. Pinned exactly so that
#: the debt neither grows nor silently outlives its fix.
PINNED_IR_DEPENDENCIES = {
    "gt4py.next.otf.binding.cpp_interface": {"gt4py.next.iterator.type_system.type_specifications"},
    "gt4py.next.otf.binding.nanobind": {"gt4py.next.iterator.type_system.type_specifications"},
}


def _is_type_checking_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_nodes(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """
    Yield the nodes that are evaluated while the module is being loaded.

    Descends through class bodies and conditionals, which execute at load
    time, but not into function bodies or `if TYPE_CHECKING:` blocks, which
    do not.
    """
    for node in nodes:
        if _is_type_checking_guard(node) or isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            continue
        yield node
        yield from _import_time_nodes(ast.iter_child_nodes(node))


def _imported_names(path: pathlib.Path, module: str) -> Iterator[str]:
    """Yield the dotted names imported when `module` is executed at import time."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in _import_time_nodes(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                parts = module.split(".")
                package = ".".join(parts[: len(parts) - (node.level - 1)])
                base = f"{package}.{node.module}" if node.module else package
            else:
                base = node.module or ""
            for alias in node.names:
                yield f"{base}.{alias.name}"


def _collect_modules() -> Mapping[str, pathlib.Path]:
    modules = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def _build_import_graph() -> Mapping[str, Set[str]]:
    """Map each module to the modules it imports at module-execution time."""
    modules = _collect_modules()
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        for name in _imported_names(path, module):
            # Resolve to the most specific existing module, so that
            # `from gt4py.next import config` is an edge to `gt4py.next.config`
            # and not to the `gt4py.next` package (whose `__init__` re-exports
            # the whole DSL and would make every edge reach everything).
            parts = name.split(".")
            while parts:
                candidate = ".".join(parts)
                if candidate in modules:
                    if candidate != module:
                        graph[module].add(candidate)
                    break
                parts.pop()
    return graph


def _is_ir_module(module: str) -> bool:
    return any(module == pkg or module.startswith(f"{pkg}.") for pkg in IR_PACKAGES)


def _shortest_path_to_ir(graph: Mapping[str, Set[str]], start: str) -> list[str] | None:
    """Return the shortest import chain from `start` into an IR package, if any."""
    queue = collections.deque([(start, [start])])
    seen = {start}
    while queue:
        module, chain = queue.popleft()
        for imported in sorted(graph.get(module, ())):
            if imported in seen:
                continue
            seen.add(imported)
            extended = [*chain, imported]
            if _is_ir_module(imported):
                return extended
            queue.append((imported, extended))
    return None


def _reachable_ir_modules(graph: Mapping[str, Set[str]], start: str) -> set[str]:
    reachable: set[str] = set()
    stack = [start]
    while stack:
        module = stack.pop()
        for imported in graph.get(module, ()):
            if imported not in reachable:
                reachable.add(imported)
                stack.append(imported)
    return {module for module in reachable if _is_ir_module(module)}


@pytest.fixture(scope="module")
def import_graph() -> Mapping[str, Set[str]]:
    return _build_import_graph()


def _expand(patterns: tuple[str, ...], graph: Mapping[str, Set[str]]) -> list[str]:
    expanded = []
    for pattern in patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-1]
            matched = sorted(module for module in graph if module.startswith(prefix))
            assert matched, f"no modules found under '{pattern}'"
            expanded.extend(matched)
        else:
            assert pattern in graph, f"unknown module '{pattern}'"
            expanded.append(pattern)
    return expanded


def test_source_root_is_the_working_tree(import_graph):
    """Guard against the check silently passing on an unexpected install layout."""
    assert "gt4py.next.otf.artifacts" in import_graph, (
        f"'{SRC_ROOT}' does not look like the gt4py source tree"
    )


def test_dsl_agnostic_otf_modules_do_not_reach_the_irs(import_graph):
    offenders = {}
    for module in _expand(IR_FREE_MODULES, import_graph):
        chain = _shortest_path_to_ir(import_graph, module)
        if chain is not None:
            offenders[module] = chain
    assert not offenders, "DSL-agnostic OTF modules must not import the GT4Py IRs:\n" + "\n".join(
        f"  {module}: " + " -> ".join(chain) for module, chain in sorted(offenders.items())
    )


@pytest.mark.parametrize("module", sorted(PINNED_IR_DEPENDENCIES))
def test_binding_ir_dependency_stays_pinned(import_graph, module):
    expected = PINNED_IR_DEPENDENCIES[module]
    actual = _reachable_ir_modules(import_graph, module)
    if actual == expected:
        return
    if not actual:
        pytest.fail(
            f"'{module}' no longer reaches the IRs. Move it to `IR_FREE_MODULES` "
            f"and drop it from `PINNED_IR_DEPENDENCIES`."
        )
    chain = _shortest_path_to_ir(import_graph, module)
    pytest.fail(
        f"'{module}' IR dependency changed.\n"
        f"  expected: {sorted(expected)}\n"
        f"  actual:   {sorted(actual)}\n"
        f"  shortest chain: {' -> '.join(chain or [])}"
    )
