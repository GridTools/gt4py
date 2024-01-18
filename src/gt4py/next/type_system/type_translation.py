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

import builtins
import collections.abc
import functools
import types
import typing
from typing import Any, ForwardRef, Optional, Union

import numpy as np
import numpy.typing as npt

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.type_system import type_info, type_specifications as ts


def get_scalar_kind(dtype: npt.DTypeLike) -> ts.ScalarKind:
    # make int & float precision platform independent.
    if dtype is builtins.int:
        dt = np.dtype("int64")
    elif dtype is builtins.float:
        dt = np.dtype("float64")
    else:
        try:
            dt = np.dtype(dtype)
        except TypeError as err:
            raise ValueError(f"Invalid scalar type definition ('{dtype}').") from err

    if dt.shape == () and dt.fields is None:
        match dt:
            case np.bool_:
                return ts.ScalarKind.BOOL
            case np.int32:
                return ts.ScalarKind.INT32
            case np.int64:
                return ts.ScalarKind.INT64
            case np.float32:
                return ts.ScalarKind.FLOAT32
            case np.float64:
                return ts.ScalarKind.FLOAT64
            case np.str_:
                return ts.ScalarKind.STRING
            case _:
                raise ValueError(f"Impossible to map '{dtype}' value to a 'ScalarKind'.")
    else:
        raise ValueError(f"Non-trivial dtypes like '{dtype}' are not yet supported.")


def from_type_hint(
    type_hint: Any,
    *,
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> ts.TypeSpec:
    recursive_make_symbol = functools.partial(from_type_hint, globalns=globalns, localns=localns)
    extra_args = ()

    # ForwardRef
    if isinstance(type_hint, str):
        type_hint = ForwardRef(type_hint)
    if isinstance(type_hint, ForwardRef):
        try:
            type_hint = xtyping.eval_forward_ref(type_hint, globalns=globalns, localns=localns)
        except Exception as error:
            raise ValueError(
                f"Type annotation '{type_hint}' has undefined forward references."
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
            return ts.ScalarType(kind=get_scalar_kind(type_hint))

        case builtins.tuple:
            if not args:
                raise ValueError(f"Tuple annotation '{type_hint}' requires at least one argument.")
            if Ellipsis in args:
                raise ValueError(f"Unbound tuples '{type_hint}' are not allowed.")
            return ts.TupleType(types=[recursive_make_symbol(arg) for arg in args])

        case common.Field:
            if (n_args := len(args)) != 2:
                raise ValueError(f"Field type requires two arguments, got {n_args}: '{type_hint}'.")

            dims: Union[Ellipsis, list[common.Dimension]] = []
            dim_arg, dtype_arg = args
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise ValueError(f"Invalid field dimension definition '{d}'.")
                    dims.append(d)
            elif dim_arg is Ellipsis:
                dims = dim_arg
            else:
                raise ValueError(f"Invalid field dimensions '{dim_arg}'.")

            try:
                dtype = recursive_make_symbol(dtype_arg)
            except ValueError as error:
                raise ValueError(
                    f"Field dtype argument must be a scalar type (got '{dtype_arg}')."
                ) from error
            if not isinstance(dtype, ts.ScalarType) or dtype.kind == ts.ScalarKind.STRING:
                raise ValueError("Field dtype argument must be a scalar type (got '{dtype}').")
            return ts.FieldType(dims=dims, dtype=dtype)

        case collections.abc.Callable:
            if not args:
                raise ValueError("Unannotated functions are not supported.")

            try:
                arg_types, return_type = args
                args = [recursive_make_symbol(arg) for arg in arg_types]
            except Exception as error:
                raise ValueError(f"Invalid callable annotations in '{type_hint}'.") from error

            kwargs_info = [arg for arg in extra_args if isinstance(arg, xtyping.CallableKwargsInfo)]
            if len(kwargs_info) != 1:
                raise ValueError(f"Invalid callable annotations in '{type_hint}'.")
            kwargs = {
                arg: recursive_make_symbol(arg_type)
                for arg, arg_type in kwargs_info[0].data.items()
            }

            # TODO(tehrengruber): print better error when no return type annotation is given
            return ts.FunctionType(
                pos_only_args=args,
                pos_or_kw_args=kwargs,
                kw_only_args={},  # TODO
                returns=recursive_make_symbol(return_type),
            )

    raise ValueError(f"'{type_hint}' type is not supported.")


def from_value(value: Any) -> ts.TypeSpec:
    # TODO(tehrengruber): use protocol from gt4py.next.common when available
    #  instead of importing from the embedded implementation
    """Make a symbol node from a Python value."""
    # TODO(tehrengruber): What we expect here currently is a GTCallable. Maybe
    #  we should check for the protocol in the future?
    if hasattr(value, "__gt_type__"):
        symbol_type = value.__gt_type__()
    elif isinstance(value, int) and not isinstance(value, bool):
        symbol_type = None
        for candidate_type in [
            ts.ScalarType(kind=ts.ScalarKind.INT32),
            ts.ScalarType(kind=ts.ScalarKind.INT64),
        ]:
            min_val, max_val = type_info.arithmetic_bounds(candidate_type)
            if min_val <= value <= max_val:
                symbol_type = candidate_type
                break
        if not symbol_type:
            raise ValueError(
                f"Value '{value}' is out of range to be representable as 'INT32' or 'INT64'."
            )
        return candidate_type
    elif isinstance(value, common.Dimension):
        symbol_type = ts.DimensionType(dim=value)
    elif common.is_field(value):
        dims = list(value.domain.dims)
        dtype = from_type_hint(value.dtype.scalar_type)
        symbol_type = ts.FieldType(dims=dims, dtype=dtype)
    elif isinstance(value, tuple):
        # Since the elements of the tuple might be one of the special cases
        # above, we can not resort to generic `infer_type` but need to do it
        # manually here. If we get rid of all the special cases this is
        # not needed anymore.
        return ts.TupleType(types=[from_value(el) for el in value])
    else:
        type_ = xtyping.infer_type(value, annotate_callable_kwargs=True)
        symbol_type = from_type_hint(type_)

    if isinstance(symbol_type, (ts.DataType, ts.CallableType, ts.OffsetType, ts.DimensionType)):
        return symbol_type
    else:
        raise ValueError(f"Impossible to map '{value}' value to a 'Symbol'.")
