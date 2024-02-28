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
from typing import Callable, TypeVar

from gt4py.next.ffront import type_info as ti_ffront
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def _expr_hash(expr: itir.Expr | str) -> str:
    """Small utility function that returns a string hash of an expression."""
    return str(abs(hash(expr)) % (10**12)).zfill(12)


def to_tuples_of_iterator(expr: itir.Expr | str, arg_type: ts.TypeSpec):
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
    param = f"__toi_{_expr_hash(expr)}"

    def fun(primitive_type, path):
        inner_expr = im.deref("it")
        for path_part in path:
            inner_expr = im.tuple_get(path_part, inner_expr)

        return im.lift(im.lambda_("it")(inner_expr))(param)

    return im.let(param, expr)(
        type_info.apply_to_primitive_constituents(
            arg_type, fun, with_path_arg=True, tuple_constructor=im.make_tuple
        )
    )


def to_iterator_of_tuples(expr: itir.Expr | str, arg_type: ts.TypeSpec):
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
    param = f"__iot_{_expr_hash(expr)}"

    type_constituents = [
        ti_ffront.promote_scalars_to_zero_dim_field(type_)
        for type_ in type_info.primitive_constituents(arg_type)
    ]
    assert all(
        isinstance(type_, ts.FieldType) and type_.dims == type_constituents[0].dims  # type: ignore[attr-defined]  # ensure by assert above
        for type_ in type_constituents
    )

    def fun(_, path):
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
        arg_type, fun, with_path_arg=True, tuple_constructor=im.make_tuple
    )
    return im.let(param, expr)(im.lift(im.lambda_(*lift_params)(stencil_expr))(*lift_args))


# TODO(tehrengruber): The code quality of this function is poor. We should rewrite it.
def process_elements(
    process_func: Callable[..., itir.Expr],
    objs: itir.Expr | list[itir.Expr],
    current_el_type: ts.TypeSpec,
):
    """
    Recursively applies a processing function to all primitive constituents of a tuple.

    Arguments:
        process_func: A callable that takes an itir.Expr representing a leaf-element of `objs`.
            If multiple `objs` are given the callable takes equally many arguments.
        objs: The object whose elements are to be transformed.
        current_el_type: A type with the same structure as the elements of `objs`. The leaf-types
            are not used and thus not relevant.
    """
    if isinstance(objs, itir.Expr):
        objs = [objs]

    _current_el_exprs = [im.ref(f"__val_{_expr_hash(obj)}") for i, obj in enumerate(objs)]
    body = _process_elements_impl(process_func, _current_el_exprs, current_el_type)

    return im.let(*((f"__val_{_expr_hash(obj)}", obj) for i, obj in enumerate(objs)))(  # type: ignore[arg-type]  # mypy not smart enough
        body
    )


T = TypeVar("T", bound=itir.Expr, covariant=True)


def _process_elements_impl(
    process_func: Callable[..., itir.Expr],
    _current_el_exprs: list[T],
    current_el_type: ts.TypeSpec,
):
    if isinstance(current_el_type, ts.TupleType):
        result = im.make_tuple(*[
            _process_elements_impl(
                process_func,
                [im.tuple_get(i, current_el_expr) for current_el_expr in _current_el_exprs],
                current_el_type.types[i],
            )
            for i in range(len(current_el_type.types))
        ])
    elif type_info.contains_local_field(current_el_type):
        raise NotImplementedError("Processing fields with local dimension is not implemented.")
    else:
        result = process_func(*_current_el_exprs)

    return result
