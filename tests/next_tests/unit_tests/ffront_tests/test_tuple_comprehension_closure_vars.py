# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Closure variable resolution inside tuple comprehension bodies.

A generator expression compiles to its own code object, so names referenced only
inside it appear in that object's `co_names` and never in the enclosing function's.
Collecting closure variables from the enclosing function alone therefore misses
them; `get_closure_vars_from_function` takes the global names from the compiler's
symbol table of the source instead, which covers nested scopes.

Free variables need no such treatment: the enclosing code object carries a cell
for them, which is why comprehensions over locally defined helpers always worked.
"""

import pytest

import gt4py.next as gtx
from gt4py.next import Dims, Dimension, float64, neighbor_sum
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.ffront.source_utils import get_closure_vars_from_function


Cell = Dimension("Cell")
Edge = Dimension("Edge")
C2EDim = Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

CField = gtx.Field[Dims[Cell], float64]
EField = gtx.Field[Dims[Edge], float64]


@gtx.field_operator
def scale(f: CField, factor: float64) -> CField:
    return f * factor


def _builtin_and_offset(tracers: tuple[EField, ...]) -> tuple[CField, ...]:
    return tuple(neighbor_sum(t(C2E), axis=C2EDim) for t in tracers)


def _module_level_operator(tracers: tuple[CField, ...], factor: float64) -> tuple[CField, ...]:
    return tuple(scale(t, factor) for t in tracers)


@pytest.mark.parametrize(
    "func", [_builtin_and_offset, _module_level_operator], ids=["builtin_and_offset", "operator"]
)
def test_module_level_name_used_only_in_comprehension(func):
    """Module-level names are reachable from inside a comprehension body."""
    parsed = FieldOperatorParser.apply_to_function(func)
    assert parsed.type is not None


def test_names_are_collected_from_the_nested_code_object():
    """The names live in the generator expression's code object, not the function's."""
    assert "neighbor_sum" not in _builtin_and_offset.__code__.co_names
    nested = [c for c in _builtin_and_offset.__code__.co_consts if hasattr(c, "co_names")]
    assert any("neighbor_sum" in c.co_names for c in nested)

    collected = get_closure_vars_from_function(_builtin_and_offset)
    assert {"neighbor_sum", "C2E", "C2EDim"} <= set(collected)


def test_comprehension_target_is_not_collected():
    """The loop target is a local of the nested code object, not a global reference."""
    assert "t" not in get_closure_vars_from_function(_module_level_operator)


def test_free_variables_still_resolve():
    """The path that already worked, kept as a guard."""

    @gtx.field_operator
    def local_scale(f: CField, factor: float64) -> CField:
        return f * factor

    def uses_freevar(tracers: tuple[CField, ...], factor: float64) -> tuple[CField, ...]:
        return tuple(local_scale(t, factor) for t in tracers)

    parsed = FieldOperatorParser.apply_to_function(uses_freevar)
    assert parsed.type is not None


def test_local_name_shadowing_a_global_is_not_collected_as_global():
    """A comprehension referencing an enclosing local must bind the local, not the global."""

    def shadows(tracers: tuple[CField, ...], factor: float64) -> tuple[CField, ...]:
        scale = local_helper  # noqa: F841  shadows the module-level 'scale'
        return tuple(scale(t, factor) for t in tracers)

    def local_helper(t, factor):
        return t

    assert "scale" not in get_closure_vars_from_function(shadows)
