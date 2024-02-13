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

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


def to_tuples_of_iterator(expr: itir.Expr | str, arg_type: ts.TypeSpec):
    """
    Convert iterator of tuples into tuples of iterator.

    >>> _ = to_tuples_of_iterator("arg", ts.TupleType(types=[ts.FieldType(dims=[],
    ...   dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))]))
    """
    param = f"__toi_{abs(hash(expr))}"

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

    >>> _ = to_iterator_of_tuples("arg", ts.TupleType(types=[ts.FieldType(dims=[],
    ...   dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))]))
    """
    param = f"__iot_{abs(hash(expr))}"

    assert all(
        isinstance(type_, ts.FieldType) for type_ in type_info.primitive_constituents(arg_type)
    )
    type_constituents = list(type_info.primitive_constituents(arg_type))
    assert all(type_.dims == type_constituents[0].dims for type_ in type_constituents)  # type: ignore[attr-defined]  # ensure by assert above

    def fun(primitive_type, path):
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
