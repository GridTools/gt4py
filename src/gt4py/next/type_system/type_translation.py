# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import collections.abc
import dataclasses
import functools
import types
import typing
from typing import Any, ForwardRef, Optional

import numpy as np
import numpy.typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.type_system import type_info, type_specifications as ts


def get_scalar_kind(dtype: npt.DTypeLike) -> ts.ScalarKind:
    # make int & float precision platform independent.
    dt: np.dtype
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
            case np.str_:
                return ts.ScalarKind.STRING
            case np.dtype():
                return getattr(ts.ScalarKind, dt.name.upper())
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
    extra_args: list = []

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
        if not isinstance(
            type_hint,
            collections.abc.Callable,  # type:ignore[arg-type] # see https://github.com/python/mypy/issues/14928
        ):
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
            tuple_types = [recursive_make_symbol(arg) for arg in args]
            assert all(isinstance(elem, ts.DataType) for elem in tuple_types)
            return ts.TupleType(types=tuple_types)  # type: ignore[arg-type] # checked in assert

        case common.Field:
            if (n_args := len(args)) != 2:
                raise ValueError(f"Field type requires two arguments, got {n_args}: '{type_hint}'.")
            dims: list[common.Dimension] = []
            dim_arg, dtype_arg = args
            dim_arg = (
                list(typing.get_args(dim_arg))
                if typing.get_origin(dim_arg) is common.Dims
                else dim_arg
            )
            if isinstance(dim_arg, list):
                for d in dim_arg:
                    if not isinstance(d, common.Dimension):
                        raise ValueError(f"Invalid field dimension definition '{d}'.")
                    dims.append(d)
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
                new_args = [recursive_make_symbol(arg) for arg in arg_types]
                assert all(isinstance(arg, ts.DataType) for arg in new_args)
            except Exception as error:
                raise ValueError(f"Invalid callable annotations in '{type_hint}'.") from error

            kwargs_info = [arg for arg in extra_args if isinstance(arg, xtyping.CallableKwargsInfo)]
            if len(kwargs_info) != 1:
                raise ValueError(f"Invalid callable annotations in '{type_hint}'.")
            kwargs = {
                arg: recursive_make_symbol(arg_type)
                for arg, arg_type in kwargs_info[0].data.items()
            }
            assert all(isinstance(val, (ts.DataType, ts.DeferredType)) for val in kwargs.values())

            returns = recursive_make_symbol(return_type)
            assert isinstance(returns, (ts.DataType, ts.DeferredType, ts.VoidType))

            # TODO(tehrengruber): print better error when no return type annotation is given
            return ts.FunctionType(
                pos_only_args=new_args,
                pos_or_kw_args=kwargs,
                kw_only_args={},  # TODO
                returns=returns,
            )
    raise ValueError(f"'{type_hint}' type is not supported.")


@dataclasses.dataclass(frozen=True)
class UnknownPythonObject(ts.TypeSpec):
    _object: Any

    def __getattr__(self, key: str) -> ts.TypeSpec:
        value = getattr(self._object, key)
        return from_value(value)

    def __deepcopy__(self, _: dict[int, Any]) -> UnknownPythonObject:
        return UnknownPythonObject(self._object)  # don't deep copy the module


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
    elif isinstance(value, common.Field):
        dims = list(value.domain.dims)
        dtype = from_type_hint(value.dtype.scalar_type)
        assert isinstance(dtype, ts.ScalarType)
        symbol_type = ts.FieldType(dims=dims, dtype=dtype)
    elif isinstance(value, tuple):
        # Since the elements of the tuple might be one of the special cases
        # above, we can not resort to generic `infer_type` but need to do it
        # manually here. If we get rid of all the special cases this is
        # not needed anymore.
        elems = [from_value(el) for el in value]
        assert all(isinstance(elem, ts.DataType) for elem in elems)
        return ts.TupleType(types=elems)  # type: ignore[arg-type] # checked in assert
    elif isinstance(value, types.ModuleType):
        return UnknownPythonObject(_object=value)
    else:
        type_ = xtyping.infer_type(value, annotate_callable_kwargs=True)
        symbol_type = from_type_hint(type_)

    if isinstance(symbol_type, (ts.DataType, ts.OffsetType, ts.DimensionType)) or (
        isinstance(symbol_type, ts.CallableType) and isinstance(symbol_type, ts.TypeSpec)
    ):
        return symbol_type
    else:
        raise ValueError(f"Impossible to map '{value}' value to a 'Symbol'.")


def as_dtype(type_: ts.ScalarType) -> core_defs.DType:
    """
    Translate a `ts.ScalarType` to a `core_defs.DType`

    >>> as_dtype(ts.ScalarType(kind=ts.ScalarKind.BOOL))  # doctest:+ELLIPSIS
    BoolDType(...)
    """
    if type_.kind == ts.ScalarKind.BOOL:
        return core_defs.BoolDType()
    elif type_.kind == ts.ScalarKind.INT32:
        return core_defs.Int32DType()
    elif type_.kind == ts.ScalarKind.INT64:
        return core_defs.Int64DType()
    elif type_.kind == ts.ScalarKind.FLOAT32:
        return core_defs.Float32DType()
    elif type_.kind == ts.ScalarKind.FLOAT64:
        return core_defs.Float64DType()
    raise ValueError(f"Scalar type '{type_}' not supported.")


def from_dtype(dtype: core_defs.DType) -> ts.ScalarType:
    """
    Translate a `core_defs.DType` to a `ts.ScalarType`

    >>> from_dtype(core_defs.BoolDType())  # doctest:+ELLIPSIS
    ScalarType(kind=...BOOL...)
    """
    if dtype == core_defs.BoolDType():
        return ts.ScalarType(kind=ts.ScalarKind.BOOL)
    elif dtype == core_defs.Int32DType():
        return ts.ScalarType(kind=ts.ScalarKind.INT32)
    elif dtype == core_defs.Int64DType():
        return ts.ScalarType(kind=ts.ScalarKind.INT64)
    elif dtype == core_defs.Float32DType():
        return ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
    elif dtype == core_defs.Float64DType():
        return ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    raise ValueError(f"DType '{dtype}' not supported.")
