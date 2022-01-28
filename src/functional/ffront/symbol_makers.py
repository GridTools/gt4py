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
from functional.ffront import common_types
from functional.ffront import field_operator_ast as foast
from functional.iterator import runtime


def make_scalar_kind(dtype: npt.DTypeLike) -> common_types.ScalarKind:
    try:
        dt = np.dtype(dtype)
    except TypeError as err:
        raise common.GTTypeError(f"Invalid scalar type definition ({dtype})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return common_types.ScalarKind.BOOL
            case np.int32:
                return common_types.ScalarKind.INT32
            case np.int64:
                return common_types.ScalarKind.INT64
            case np.float32:
                return common_types.ScalarKind.FLOAT32
            case np.float64:
                return common_types.ScalarKind.FLOAT64
            case _:
                raise common.GTTypeError(f"Impossible to map '{dtype}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{dtype}' are not yet supported")


def make_symbol_type_from_typing(
    type_hint: Any,
    *,
    global_ns: Optional[dict[str, Any]] = None,
    local_ns: Optional[dict[str, Any]] = None,
) -> common_types.SymbolType:
    recursive_make_symbol = functools.partial(
        make_symbol_type_from_typing, global_ns=global_ns, local_ns=local_ns
    )
    extra_args = ()

    # ForwardRef
    if isinstance(type_hint, str):
        type_hint = ForwardRef(type_hint)
    if isinstance(type_hint, ForwardRef):
        try:
            type_hint = typingx.resolve_type(
                type_hint, global_ns=global_ns, local_ns=local_ns, allow_partial=False
            )
        except Exception as error:
            raise FieldOperatorTypeError(
                f"Type annotation ({type_hint}) has undefined forward references!"
            ) from error

    # Annotated
    if typing.get_origin(type_hint) is typing.Annotated:
        type_hint, *extra_args = typing.get_args(type_hint)
        if not isinstance(type_hint, collections.abc.Callable):
            type_hint = typingx.resolve_type(
                type_hint, global_ns=global_ns, local_ns=local_ns, allow_partial=False
            )

    canonical_type = (
        typing.get_origin(type_hint)
        if isinstance(type_hint, types.GenericAlias) or type(type_hint).__module__ == "typing"
        else type_hint
    )
    args = typing.get_args(type_hint)

    match canonical_type:
        case type() as t if issubclass(t, (bool, int, float, np.generic)):
            return common_types.ScalarType(kind=make_scalar_kind(type_hint))

        case builtins.tuple:
            if not args:
                raise FieldOperatorTypeError(
                    f"Tuple annotation ({type_hint}) requires at least one argument!"
                )
            if Ellipsis in args:
                raise FieldOperatorTypeError(f"Unbound tuples ({type_hint}) are not allowed!")
            return common_types.TupleType(types=[recursive_make_symbol(arg) for arg in args])

        case common.Field:
            if (n_args := len(args)) != 2:
                raise FieldOperatorTypeError(
                    f"Field type requires two arguments, got {n_args}! ({type_hint})"
                )

            dims: Union[Ellipsis, list[common.Dimension]] = []
            dim_arg, dtype_arg = args
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise FieldOperatorTypeError(f"Invalid field dimension definition '{d}'")
                    dims.append(d)
            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise FieldOperatorTypeError(f"Invalid field dimensions '{dim_arg}'")

            try:
                dtype = recursive_make_symbol(dtype_arg)
            except FieldOperatorTypeError as error:
                raise FieldOperatorTypeError(
                    "Field dtype argument must be a scalar type (got '{dtype}')!"
                ) from error
            if not isinstance(dtype, common_types.ScalarType):
                raise FieldOperatorTypeError(
                    "Field dtype argument must be a scalar type (got '{dtype}')!"
                )
            return common_types.FieldType(dims=dims, dtype=dtype)

        case collections.abc.Callable:
            if not args:
                raise FieldOperatorTypeError("Not annotated functions are not supported!")

            try:
                arg_types, return_type = args
                args = [recursive_make_symbol(arg) for arg in arg_types]
            except Exception as error:
                raise FieldOperatorTypeError(
                    f"Invalid callable annotations in {type_hint}"
                ) from error

            kwargs_info = [arg for arg in extra_args if isinstance(arg, typingx.CallableKwargsInfo)]
            if len(kwargs_info) != 1:
                raise FieldOperatorTypeError(f"Invalid callable annotations in {type_hint}")
            kwargs = {
                arg: recursive_make_symbol(arg_type)
                for arg, arg_type in kwargs_info[0].data.items()
            }

            return common_types.FunctionType(
                args=args, kwargs=kwargs, returns=recursive_make_symbol(return_type)
            )
        case runtime.Offset:
            return common_types.OffsetType()

    raise FieldOperatorTypeError(f"'{type_hint}' type is not supported")


def make_symbol_from_value(
    name: str, value: Any, namespace: foast.Namespace, location: SourceLocation
) -> foast.Symbol:
    if not isinstance(value, type) or type(value).__module__ != "typing":
        value = typingx.get_typing(value, annotate_callable_kwargs=True)

    symbol_type = make_symbol_type_from_typing(value)

    if isinstance(symbol_type, common_types.DataType):
        return foast.DataSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
    elif isinstance(symbol_type, common_types.FunctionType):
        return foast.Function(
            id=name,
            type=symbol_type,
            namespace=namespace,
            location=location,
        )
    elif isinstance(symbol_type, common_types.OffsetType):
        return foast.OffsetSymbol(id=name, type=symbol_type, namespace=namespace, location=location)
    else:
        raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


# TODO(egparedes): Add source location info (maybe subclassing FieldOperatorSyntaxError)
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
