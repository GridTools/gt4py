# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

import builtins
import collections.abc
import functools
import types
import typing
from typing import Any, ForwardRef, Optional, Union

import numpy as np
import numpy.typing as npt

from eve import typingx
from eve.type_definitions import SourceLocation
from functional import common
from functional.ffront import field_operator_ast as foast


def make_scalar_kind(dtype: npt.DTypeLike) -> foast.ScalarKind:
    try:
        dt = np.dtype(dtype)
    except TypeError as err:
        raise common.GTTypeError(f"Invalid scalar type definition ({dtype})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return foast.ScalarKind.BOOL
            case np.int32:
                return foast.ScalarKind.INT32
            case np.int64:
                return foast.ScalarKind.INT64
            case np.float32:
                return foast.ScalarKind.FLOAT32
            case np.float64:
                return foast.ScalarKind.FLOAT64
            case _:
                raise common.GTTypeError(f"Impossible to map '{dtype}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{dtype}' are not yet supported")


def make_symbol_type_from_typing(
    value: Any,
    *,
    global_ns: Optional[dict[str, Any]] = None,
    local_ns: Optional[dict[str, Any]] = None,
) -> foast.SymbolType:
    recursive_make_symbol = functools.partial(
        make_symbol_type_from_typing, global_ns=global_ns, local_ns=local_ns
    )
    extra_args = ()

    # ForwardRef
    if isinstance(value, str):
        value = ForwardRef(value)
    if isinstance(value, ForwardRef):
        try:
            value = typingx.resolve_type(
                value, global_ns=global_ns, local_ns=local_ns, allow_partial=False
            )
        except Exception as error:
            raise FieldOperatorTypeError(
                f"Type annotation ({value}) has undefined forward references!"
            ) from error

    # Annotated
    if typing.get_origin(value) is typing.Annotated:
        value, *extra_args = typing.get_args(value)
        if not isinstance(value, collections.abc.Callable):
            value = typingx.resolve_type(
                value, global_ns=global_ns, local_ns=local_ns, allow_partial=False
            )

    value_type = (
        typing.get_origin(value)
        if isinstance(value, types.GenericAlias) or type(value).__module__ == "typing"
        else value
    )
    args = typing.get_args(value)

    match value_type:
        case type() as t if issubclass(t, (bool, int, float, np.generic)):
            return foast.ScalarType(kind=make_scalar_kind(value))

        case builtins.tuple:
            if not args:
                raise FieldOperatorTypeError(
                    f"Tuple annotation ({value}) requires at least one argument!"
                )
            if Ellipsis in args:
                raise FieldOperatorTypeError(
                    f"Unbound tuples ({value}) are not allowed!"
                )
            return foast.TupleType(types=[recursive_make_symbol(arg) for arg in args])

        case common.Field:
            if len(args) != 2:
                raise FieldOperatorTypeError(f"Field type requires two arguments, got {len(args)}!")

            dims: Union[Ellipsis, list[foast.Dimension]] = []
            dim_arg, dtype_arg = args
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise FieldOperatorTypeError(f"Invalid field dimension definition '{d}'")
                    dims.append(foast.Dimension(name=d.value))
            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise FieldOperatorTypeError(f"Invalid field type dimensions '{dim_arg}'")

            dtype = recursive_make_symbol(dtype_arg)
            if not isinstance(dtype, foast.ScalarType):
                raise FieldOperatorTypeError(
                    "Field type dtype argument must be a scalar type (got '{dtype}')!"
                )
            return foast.FieldType(dims=dims, dtype=dtype)

        case collections.abc.Callable:
            arg_types, return_type = args
            if arg_types in (None, Ellipsis) or return_type is None:
                raise FieldOperatorTypeError("Not annotated functions are not supported!")

            kwargs_info = [arg for arg in extra_args if isinstance(arg, typingx.CallableKwargsInfo)]
            if len(kwargs_info) != 1:
                raise FieldOperatorTypeError(f"Invalid callable annotations in {value}")
            kwargs_info = kwargs_info[0]

            return foast.FunctionType(
                args=[recursive_make_symbol(arg) for arg in arg_types],
                kwargs={
                    arg: recursive_make_symbol(arg_type)
                    for arg, arg_type in kwargs_info.data.items()
                },
                returns=recursive_make_symbol(return_type),
            )

    raise FieldOperatorTypeError(f"'{value}' type is not supported")


def make_symbol_from_value(
    name: str, value: Any, namespace: foast.Namespace, location: SourceLocation
) -> foast.Symbol:
    if not isinstance(value, type) or type(value).__module__ != "typing":
        value = typingx.get_typing(value, annotate_callable_kwargs=True)
    symbol_type = make_symbol_type_from_typing(value)
    if isinstance(symbol_type, foast.DataType):
        return foast.DataSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
    elif isinstance(symbol_type, foast.FunctionType):
        return foast.Function(
            id=name,
            type=symbol_type,
            namespace=namespace,
            location=location,
        )
    else:
        raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


class FieldOperatorTypeError(common.GTTypeError):
    def __init__(
        self,
        msg="",
        *,
        info=None,
    ):
        msg = f"Invalid type declaration: {msg}"
        args = tuple([msg, info] if info else [msg])
        super().__init__(*args)
