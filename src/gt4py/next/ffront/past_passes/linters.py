# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterator
from typing import Any

from gt4py.eve import SourceLocation
from gt4py.next import errors
from gt4py.next.ffront import (
    fbuiltins,
    gtcallable,
    program_ast as past,
    stages as ffront_stages,
    transform_utils,
    type_info as ffront_type_info,  # noqa: F401 [unused-import]  # registers 'canonicalize_arguments'
)
from gt4py.next.ffront.stages import ConcretePASTProgramDef, PASTProgramDef
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import check_inout_field
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


def lint_misnamed_functions(
    inp: ffront_stages.PASTProgramDef,
) -> ffront_stages.PASTProgramDef:
    function_closure_vars = transform_utils._filter_closure_vars_by_type(
        inp.closure_vars, gtcallable.GTCallable
    )
    misnamed_functions = [
        f"{name} vs. {func.__gt_itir__().id}"
        for name, func in function_closure_vars.items()
        if name != func.__gt_itir__().id
    ]
    if misnamed_functions:
        raise RuntimeError(
            f"The following symbols resolve to a function with a mismatching name: {','.join(misnamed_functions)}."
        )
    return inp


def lint_undefined_symbols(
    inp: ffront_stages.PASTProgramDef,
) -> ffront_stages.PASTProgramDef:
    undefined_symbols = [
        symbol.id for symbol in inp.past_node.closure_vars if symbol.id not in inp.closure_vars
    ]
    if undefined_symbols:
        raise RuntimeError(
            f"The following closure variables are undefined: {', '.join(undefined_symbols)}."
        )
    return inp


def _field_view_expr(expr: past.Expr) -> itir.Expr:
    """Lower a call argument or 'out' target, as far as needed to see which fields it refers to."""
    match expr:
        case past.Name():
            return im.ref(expr.id)
        case past.Subscript():  # slicing does not change which elements are read or written
            return _field_view_expr(expr.value)
        case past.TupleExpr():
            return im.make_tuple(*(_field_view_expr(el) for el in expr.elts))
        case past.Attribute(value=past.Expr(type=ts.NamedCollectionType() as type_)):
            return im.tuple_get(type_.keys.index(expr.attr), _field_view_expr(expr.value))
        case _:
            # a scalar expression: it reads no field, so any expression without references does
            return im.literal_from_value(0)


def _field_refs(expr: past.Expr) -> Iterator[tuple[str, tuple[int, ...], SourceLocation]]:
    """Yield the (elements of) parameters `expr` refers to, with the location of the reference."""
    match expr:
        case past.Name():
            yield expr.id, (), expr.location
        case past.Subscript():
            for name, path, _ in _field_refs(expr.value):
                yield name, path, expr.location
        case past.TupleExpr():
            for el in expr.elts:
                yield from _field_refs(el)
        case past.Attribute(value=past.Expr(type=ts.NamedCollectionType() as type_)):
            for name, path, _ in _field_refs(expr.value):
                yield name, (*path, type_.keys.index(expr.attr)), expr.location


def _overlaps(path: tuple[int, ...], other: tuple[int, ...]) -> bool:
    return path[: len(other)] == other[: len(path)]


def _lowered_closure(
    inp: ffront_stages.PASTProgramDef,
) -> tuple[list[itir.FunctionDefinition], dict[str, ts.OffsetType]]:
    """The function definitions and offset types the statements of `inp` can refer to."""
    all_closure_vars = transform_utils._get_closure_vars_recursively(inp.closure_vars)
    function_definitions = [
        func.__gt_gtir__()
        for func in transform_utils._filter_closure_vars_by_type(
            all_closure_vars, gtcallable.GTCallable
        ).values()
    ]
    offset_types = {
        offset.value: offset.__gt_type__()
        for offset in transform_utils._filter_closure_vars_by_type(
            all_closure_vars, fbuiltins.FieldOffset
        ).values()
    }
    return function_definitions, offset_types


def lint_inout_fields_read_with_offset(
    inp: ffront_stages.PASTProgramDef,
) -> ffront_stages.PASTProgramDef:
    """
    Reject calls that write a field which they also read at another point.

    A program statement may only read a field it also writes at the point being written, see
    ADR 18, rule 3, and `check_inout_field.find_inout_fields_read_with_offset` for the details.
    Checking this on the field-view program makes the error independent of the backend.
    """
    closure = None
    for stmt in inp.past_node.body:
        assert isinstance(stmt.func.type, ts.CallableType)
        out = stmt.kwargs["out"]
        call_kwargs = {k: v for k, v in stmt.kwargs.items() if k not in ("out", "domain")}
        call_args = [*stmt.args, *call_kwargs.values()]
        written = {name for name, _, _ in _field_refs(out)}
        if not any(name in written for arg in call_args for name, _, _ in _field_refs(arg)):
            continue  # the call does not read what it writes, the common case

        if closure is None:  # lowering is not free, only do it if needed
            closure = _lowered_closure(inp)
        function_definitions, offset_types = closure
        args, kwargs = type_info.canonicalize_arguments(stmt.func.type, stmt.args, call_kwargs)
        expr = im.call(stmt.func.id)(*(_field_view_expr(arg) for arg in [*args, *kwargs.values()]))

        violations = check_inout_field.find_inout_fields_read_with_offset(
            expr,
            _field_view_expr(out),
            function_definitions=function_definitions,
            offset_types=offset_types,
        )
        if violations:
            violation = violations[0]
            location = next(
                (
                    loc
                    for name, path, loc in _field_refs(out)
                    if name == violation.name and _overlaps(path, violation.path)
                ),
                out.location,
            )
            related = [
                (loc, "read here")
                for arg in call_args
                for name, path, loc in _field_refs(arg)
                if name == violation.name and _overlaps(path, violation.path)
            ]
            raise errors.InOutFieldReadWithOffsetError(
                location, violation.field, violation.offsets, related=related
            )
    return inp


def linter_factory(
    cached: bool = True, adapter: bool = True
) -> workflow.Workflow[PASTProgramDef, PASTProgramDef]:
    wf: workflow.Workflow[PASTProgramDef, PASTProgramDef] = workflow.StepSequence(
        (lint_misnamed_functions, lint_undefined_symbols, lint_inout_fields_read_with_offset)
    )
    if cached:
        wf = workflow.CachedStep.in_memory(
            step=wf, input_fingerprinter=ffront_stages.semantic_fingerprinter
        )
    return wf


def adapted_linter_factory(
    **kwargs: Any,
) -> workflow.Workflow[ConcretePASTProgramDef, ConcretePASTProgramDef]:
    return toolchain.DataOnlyAdapter(linter_factory(**kwargs))
