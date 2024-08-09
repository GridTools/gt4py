# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

from gt4py.eve import utils as eve_utils
from gt4py.next import utils as next_utils
from gt4py.next.ffront import type_info as ti_ffront
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def to_tuples_of_iterator(expr: itir.Expr | str, arg_type: ts.TypeSpec) -> itir.FunCall:
    """
    Convert iterator of tuples into tuples of iterator.

    Supports arbitrary nesting.

    >>> print(
    ...     to_tuples_of_iterator(
    ...         "arg",
    ...         ts.TupleType(
    ...             types=[ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))]
    ...         ),
    ...     )
    ... )  # doctest: +ELLIPSIS
    (λ(__toi_...) → {(↑(λ(it) → (·it)[0]))(__toi_...)})(arg)
    """
    param = f"__toi_{eve_utils.content_hash(expr)}"

    def fun(primitive_type: ts.TypeSpec, path: tuple[int, ...]) -> itir.Expr:
        inner_expr = im.deref("it")
        for path_part in path:
            inner_expr = im.tuple_get(path_part, inner_expr)

        return im.lift(im.lambda_("it")(inner_expr))(param)

    return im.let(param, expr)(
        type_info.apply_to_primitive_constituents(
            fun, arg_type, with_path_arg=True, tuple_constructor=im.make_tuple
        )
    )


def to_iterator_of_tuples(expr: itir.Expr | str, arg_type: ts.TypeSpec) -> itir.FunCall:
    """
    Convert tuples of iterator into iterator of tuples.

    Supports arbitrary nesting.

    >>> print(
    ...     to_iterator_of_tuples(
    ...         "arg",
    ...         ts.TupleType(
    ...             types=[ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))]
    ...         ),
    ...     )
    ... )  # doctest: +ELLIPSIS
    (λ(__iot_...) → (↑(λ(__iot_el_0) → {·__iot_el_0}))(__iot_...[0]))(arg)
    """
    param = f"__iot_{eve_utils.content_hash(expr)}"

    type_constituents = [
        ti_ffront.promote_scalars_to_zero_dim_field(type_)
        for type_ in type_info.primitive_constituents(arg_type)
    ]
    assert all(
        isinstance(type_, ts.FieldType) and type_.dims == type_constituents[0].dims  # type: ignore[attr-defined]  # ensure by assert above
        for type_ in type_constituents
    )

    def fun(_: Any, path: tuple[int, ...]) -> itir.FunCall:
        param_name = "__iot_el"
        for path_part in path:
            param_name = f"{param_name}_{path_part}"
        return im.deref(param_name)

    lift_params, lift_args = [], []
    for _, path in type_info.primitive_constituents(arg_type, with_path_arg=True):
        param_name, arg_expr = "__iot_el", param
        for path_part in path:
            param_name = f"{param_name}_{path_part}"
            arg_expr = im.tuple_get(path_part, arg_expr)

        lift_params.append(param_name)
        lift_args.append(arg_expr)

    stencil_expr = type_info.apply_to_primitive_constituents(
        fun, arg_type, with_path_arg=True, tuple_constructor=im.make_tuple
    )
    return im.let(param, expr)(im.lift(im.lambda_(*lift_params)(stencil_expr))(*lift_args))


T = TypeVar("T", bound=itir.Expr, covariant=True)


def expand_tuple_expr(tup: itir.Expr, tup_type: ts.TypeSpec) -> tuple[itir.Expr | tuple, ...]:
    """
    Create a tuple of `tuple_get` calls on `tup` by using the structure provided by `tup_type`.

    Examples:
        >>> expand_tuple_expr(
        ...     itir.SymRef(id="tup"),
        ...     ts.TupleType(
        ...         types=[
        ...             ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
        ...             ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
        ...         ]
        ...     ),
        ... )
        (FunCall(fun=SymRef(id=SymbolRef('tuple_get')), args=[Literal(value='0', type=ScalarType(kind=<ScalarKind.INT32: 32>, shape=None)), SymRef(id=SymbolRef('tup'))]), FunCall(fun=SymRef(id=SymbolRef('tuple_get')), args=[Literal(value='1', type=ScalarType(kind=<ScalarKind.INT32: 32>, shape=None)), SymRef(id=SymbolRef('tup'))]))
    """

    def _tup_get(index_and_type: tuple[int, ts.TypeSpec]) -> itir.Expr:
        i, _ = index_and_type
        return im.tuple_get(i, tup)

    res = next_utils.tree_map(collection_type=list, result_collection_type=tuple)(_tup_get)(
        next_utils.tree_enumerate(
            tup_type, collection_type=ts.TupleType, result_collection_type=list
        )
    )
    assert isinstance(res, tuple)  # for mypy
    return res


def process_elements(
    process_func: Callable[..., itir.Expr],
    objs: itir.Expr | Iterable[itir.Expr],
    current_el_type: ts.TypeSpec,
) -> itir.FunCall:
    """
    Arguments:
        process_func: A callable that takes an itir.Expr representing a leaf-element of `objs`.
            If multiple `objs` are given the callable takes equally many arguments.
        objs: The object whose elements are to be transformed.
        current_el_type: A type with the same structure as the elements of `objs`. The leaf-types
            are not used and thus not relevant.
    """
    if isinstance(objs, itir.Expr):
        objs = (objs,)
        zipper = lambda x: x
    else:
        zipper = lambda *x: x
    expanded = [expand_tuple_expr(arg, current_el_type) for arg in objs]
    tree_zip = next_utils.tree_map(result_collection_type=list)(zipper)(*expanded)
    result = next_utils.tree_map(
        collection_type=list, result_collection_type=lambda x: im.make_tuple(*x)
    )(process_func)(tree_zip)
    return result
