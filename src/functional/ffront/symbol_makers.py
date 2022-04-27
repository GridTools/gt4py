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

from eve import extended_typing as xtyping
from functional import common
from functional.ffront import common_types as ct


def make_scalar_kind(dtype: npt.DTypeLike) -> ct.ScalarKind:
    # make int & float precision platform independent.
    if dtype is builtins.int:
        dt = np.dtype("int64")
    elif dtype is builtins.float:
        dt = np.dtype("float64")
    else:
        try:
            dt = np.dtype(dtype)
        except TypeError as err:
            raise common.GTTypeError(f"Invalid scalar type definition ({dtype})") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return ct.ScalarKind.BOOL
            case np.int32:
                return ct.ScalarKind.INT32
            case np.int64:
                return ct.ScalarKind.INT64
            case np.float32:
                return ct.ScalarKind.FLOAT32
            case np.float64:
                return ct.ScalarKind.FLOAT64
            case np.str:
                return ct.ScalarKind.STRING
            case _:
                raise common.GTTypeError(f"Impossible to map '{dtype}' value to a ScalarKind")
    else:
        raise common.GTTypeError(f"Non-trivial dtypes like '{dtype}' are not yet supported")


def make_symbol_type_from_typing(
    type_hint: Any,
    *,
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> ct.SymbolType:
    recursive_make_symbol = functools.partial(
        make_symbol_type_from_typing, globalns=globalns, localns=localns
    )
    extra_args = ()

    # ForwardRef
    if isinstance(type_hint, str):
        type_hint = ForwardRef(type_hint)
    if isinstance(type_hint, ForwardRef):
        try:
            type_hint = xtyping.eval_forward_ref(type_hint, globalns=globalns, localns=localns)
        except Exception as error:
            raise TypingError(
                f"Type annotation ({type_hint}) has undefined forward references!"
            ) from error

    # Annotated
    if typing.get_origin(type_hint) is typing.Annotated:
        type_hint, *extra_args = typing.get_args(type_hint)
        if not isinstance(type_hint, collections.abc.Callable):
            type_hint = xtyping.eval_forward_ref(type_hint, globalns=globalns, localns=localns)

    canonical_type = (
        typing.get_origin(type_hint)
        if isinstance(type_hint, types.GenericAlias) or type(type_hint).__module__ == "typing"
        else type_hint
    )
    args = typing.get_args(type_hint)

    match canonical_type:
        case type() as t if issubclass(t, (bool, int, float, np.generic, str)):
            return ct.ScalarType(kind=make_scalar_kind(type_hint))

        case builtins.tuple:
            if not args:
                raise TypingError(f"Tuple annotation ({type_hint}) requires at least one argument!")
            if Ellipsis in args:
                raise TypingError(f"Unbound tuples ({type_hint}) are not allowed!")
            return ct.TupleType(types=[recursive_make_symbol(arg) for arg in args])

        case common.Field:
            if (n_args := len(args)) != 2:
                raise TypingError(f"Field type requires two arguments, got {n_args}! ({type_hint})")

            dims: Union[Ellipsis, list[common.Dimension]] = []
            dim_arg, dtype_arg = args
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise TypingError(f"Invalid field dimension definition '{d}'")
                    dims.append(d)
            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise TypingError(f"Invalid field dimensions '{dim_arg}'")

            try:
                dtype = recursive_make_symbol(dtype_arg)
            except TypingError as error:
                raise TypingError(
                    f"Field dtype argument must be a scalar type (got '{dtype_arg}')!"
                ) from error
            if not isinstance(dtype, ct.ScalarType) or dtype.kind == ct.ScalarKind.STRING:
                raise TypingError("Field dtype argument must be a scalar type (got '{dtype}')!")
            return ct.FieldType(dims=dims, dtype=dtype)

        case collections.abc.Callable:
            if not args:
                raise TypingError("Not annotated functions are not supported!")

            try:
                arg_types, return_type = args
                args = [recursive_make_symbol(arg) for arg in arg_types]
            except Exception as error:
                raise TypingError(f"Invalid callable annotations in {type_hint}") from error

            kwargs_info = [arg for arg in extra_args if isinstance(arg, xtyping.CallableKwargsInfo)]
            if len(kwargs_info) != 1:
                raise TypingError(f"Invalid callable annotations in {type_hint}")
            kwargs = {
                arg: recursive_make_symbol(arg_type)
                for arg, arg_type in kwargs_info[0].data.items()
            }

            # TODO(tehrengruber): print better error when no return type annotation is given
            return ct.FunctionType(
                args=args, kwargs=kwargs, returns=recursive_make_symbol(return_type)
            )

        case type() as t if issubclass(t, common.Dimension):
            return ct.OffsetType()

    raise TypingError(f"'{type_hint}' type is not supported")


def make_symbol_type_from_value(value: Any) -> ct.SymbolType:
    """Make a symbol node from a Python value."""
    # TODO(tehrengruber): What we expect here currently is a GTCallable. Maybe
    #  we should check for the protocol in the future?
    if hasattr(value, "__gt_type__"):
        symbol_type = value.__gt_type__()

    else:
        type_ = xtyping.reveal_type(value, annotate_callable_kwargs=True)
        symbol_type = make_symbol_type_from_typing(type_)

    if isinstance(symbol_type, (ct.DataType, ct.FunctionType, ct.OffsetType)):
        return symbol_type
    else:
        raise common.GTTypeError(f"Impossible to map '{value}' value to a Symbol")


# TODO(egparedes): Add source location info (maybe subclassing FieldOperatorSyntaxError)
class TypingError(common.GTTypeError):
    def __init__(
        self,
        msg="",
        *,
        info=None,
    ):
        msg = f"Invalid type declaration: {msg}"
        args = tuple([msg, info] if info else [msg])
        super().__init__(*args)
