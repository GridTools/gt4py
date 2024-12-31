# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any, Callable, Optional, TypeVar

from gt4py.eve import utils as eve_utils
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


# TODO(tehrengruber): The code quality of this function is poor. We should rewrite it.
def process_elements(
    process_func: Callable[..., itir.Expr],
    objs: itir.Expr | Iterable[itir.Expr],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]] = None,
) -> itir.FunCall:
    """
    Recursively applies a processing function to all primitive constituents of a tuple.

    Arguments:
        process_func: A callable that takes an itir.Expr representing a leaf-element of `objs`.
            If multiple `objs` are given the callable takes equally many arguments.
        objs: The object whose elements are to be transformed.
        current_el_type: A type with the same structure as the elements of `objs`. The leaf-types
            are not used and thus not relevant.
        arg_types: If provided, a tuple of the type of each argument is passed to `process_func` as last argument.
            Note, that `arg_types` might coincide with `(current_el_type,)*len(objs)`, but not necessarily,
            in case of implicit broadcasts.
    """
    if isinstance(objs, itir.Expr):
        objs = (objs,)

    let_ids = tuple(f"__val_{eve_utils.content_hash(obj)}" for obj in objs)
    body = _process_elements_impl(
        process_func,
        tuple(im.ref(let_id) for let_id in let_ids),
        current_el_type,
        arg_types=arg_types,
    )

    return im.let(*(zip(let_ids, objs, strict=True)))(body)


T = TypeVar("T", bound=itir.Expr, covariant=True)


def _process_elements_impl(
    process_func: Callable[..., itir.Expr],
    _current_el_exprs: Iterable[T],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]],
) -> itir.Expr:
    if isinstance(current_el_type, ts.TupleType):
        result = im.make_tuple(
            *(
                _process_elements_impl(
                    process_func,
                    tuple(
                        im.tuple_get(i, current_el_expr) for current_el_expr in _current_el_exprs
                    ),
                    current_el_type.types[i],
                    arg_types=tuple(arg_t.types[i] for arg_t in arg_types)  # type: ignore[attr-defined] # guaranteed by the requirement that `current_el_type` and each element of `arg_types` have the same tuple structure
                    if arg_types is not None
                    else None,
                )
                for i in range(len(current_el_type.types))
            )
        )
    else:
        if arg_types is not None:
            result = process_func(*_current_el_exprs, arg_types)
        else:
            result = process_func(*_current_el_exprs)

    return result
