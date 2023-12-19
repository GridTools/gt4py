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

import dataclasses
import functools
import inspect
from builtins import bool, float, int, tuple  # noqa: A004
from typing import Any, Callable, Generic, ParamSpec, Tuple, TypeAlias, TypeVar, Union, cast

import numpy as np
from numpy import float32, float64, int32, int64

import gt4py.next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import common, embedded
from gt4py.next.common import Dimension, Field  # noqa: F401  # direct import for TYPE_BUILTINS
from gt4py.next.ffront.experimental import as_offset  # noqa: F401
from gt4py.next.iterator import runtime
from gt4py.next.type_system import type_specifications as ts


PYTHON_TYPE_BUILTINS = [bool, int, float, tuple]
PYTHON_TYPE_BUILTIN_NAMES = [t.__name__ for t in PYTHON_TYPE_BUILTINS]

TYPE_BUILTINS = [
    common.Field,
    common.Dimension,
    int32,
    int64,
    float32,
    float64,
] + PYTHON_TYPE_BUILTINS
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]

# Be aware: Type aliases are not fully supported in the frontend yet, e.g. `IndexType(1)` will not
# work.
IndexType: TypeAlias = int32

TYPE_ALIAS_NAMES = ["IndexType"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _type_conversion_helper(t: type) -> type[ts.TypeSpec] | tuple[type[ts.TypeSpec], ...]:
    if t is common.Field:
        return ts.FieldType
    elif t is common.Dimension:
        return ts.DimensionType
    elif t is core_defs.ScalarT:
        return ts.ScalarType
    elif t is type:
        return (
            ts.FunctionType
        )  # our type of type is currently represented by the type constructor function
    elif t is Tuple or (hasattr(t, "__origin__") and t.__origin__ is tuple):
        return ts.TupleType
    elif hasattr(t, "__origin__") and t.__origin__ is Union:
        types = [_type_conversion_helper(e) for e in t.__args__]  # type: ignore[attr-defined]
        assert all(type(t) is type and issubclass(t, ts.TypeSpec) for t in types)
        return cast(tuple[type[ts.TypeSpec], ...], tuple(types))  # `cast` to break the recursion
    else:
        raise AssertionError("Illegal type encountered.")


def _type_conversion(t: type) -> ts.DeferredType:
    return ts.DeferredType(constraint=_type_conversion_helper(t))


@dataclasses.dataclass(frozen=True)
class BuiltInFunction(Generic[_R, _P]):
    name: str = dataclasses.field(init=False)
    # `function` can be used to provide a default implementation for all `Field` implementations,
    # e.g. a fused multiply add could have a default implementation as a*b+c, but an optimized implementation for a specific `Field`
    function: Callable[_P, _R]

    def __post_init__(self):
        object.__setattr__(self, "name", f"{self.function.__module__}.{self.function.__name__}")

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        impl = self.dispatch(*args)
        return impl(*args, **kwargs)

    def dispatch(self, *args: Any) -> Callable[_P, _R]:
        arg_types = tuple(type(arg) for arg in args)
        for atype in arg_types:
            # current strategy is to select the implementation of the first arg that supports the operation
            # TODO: define a strategy that converts or prevents conversion
            if (dispatcher := getattr(atype, "__gt_builtin_func__", None)) is not None and (
                op_func := dispatcher(self)
            ) is not NotImplemented:
                return op_func
        else:
            return self.function

    def __gt_type__(self) -> ts.FunctionType:
        signature = inspect.signature(self.function)
        params = signature.parameters

        return ts.FunctionType(
            pos_only_args=[
                _type_conversion(param.annotation)
                for param in params.values()
                if param.kind == inspect.Parameter.POSITIONAL_ONLY
            ],
            pos_or_kw_args={
                k: _type_conversion(v.annotation)
                for k, v in params.items()
                if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            },
            kw_only_args={
                k: _type_conversion(v.annotation)
                for k, v in params.items()
                if v.kind == inspect.Parameter.KEYWORD_ONLY
            },
            returns=_type_conversion(signature.return_annotation),
        )


MaskT = TypeVar("MaskT", bound=common.Field)
FieldT = TypeVar("FieldT", bound=Union[common.Field, core_defs.Scalar, Tuple])


class WhereBuiltinFunction(
    BuiltInFunction[_R, [MaskT, FieldT, FieldT]], Generic[_R, MaskT, FieldT]
):
    def __call__(self, mask: MaskT, true_field: FieldT, false_field: FieldT) -> _R:
        if isinstance(true_field, tuple) or isinstance(false_field, tuple):
            if not (isinstance(true_field, tuple) and isinstance(false_field, tuple)):
                raise ValueError(
                    # TODO(havogt) find a strategy to unify parsing and embedded error messages
                    f"Either both or none can be tuple in '{true_field=}' and '{false_field=}'."
                )
            if len(true_field) != len(false_field):
                raise ValueError(
                    "Tuple of different size not allowed."
                )  # TODO(havogt) find a strategy to unify parsing and embedded error messages
            return tuple(
                where(mask, t, f) for t, f in zip(true_field, false_field)
            )  # type: ignore[return-value] # `tuple` is not `_R`
        return super().__call__(mask, true_field, false_field)


@BuiltInFunction
def neighbor_sum(
    field: common.Field,
    /,
    axis: common.Dimension,
) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def max_over(
    field: common.Field,
    /,
    axis: common.Dimension,
) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def min_over(
    field: common.Field,
    /,
    axis: common.Dimension,
) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def broadcast(
    field: common.Field | core_defs.ScalarT,
    dims: tuple[common.Dimension, ...],
    /,
) -> common.Field:
    assert core_defs.is_scalar_type(
        field
    )  # default implementation for scalars, Fields are handled via dispatch
    return common.field(
        np.asarray(field)[
            tuple([np.newaxis] * len(dims))
        ],  # TODO(havogt) use FunctionField once available
        domain=common.Domain(dims=dims, ranges=tuple([common.UnitRange.infinite()] * len(dims))),
    )


@WhereBuiltinFunction
def where(
    mask: common.Field,
    true_field: common.Field | core_defs.ScalarT | Tuple,
    false_field: common.Field | core_defs.ScalarT | Tuple,
    /,
) -> common.Field | Tuple:
    raise NotImplementedError()


@BuiltInFunction
def astype(
    value: common.Field | core_defs.ScalarT | Tuple,
    type_: type,
    /,
) -> common.Field | core_defs.ScalarT | Tuple:
    if isinstance(value, tuple):
        return tuple(astype(v, type_) for v in value)
    # default implementation for scalars, Fields are handled via dispatch
    assert core_defs.is_scalar_type(value)
    return core_defs.dtype(type_).scalar_type(value)


UNARY_MATH_NUMBER_BUILTIN_NAMES = ["abs"]

UNARY_MATH_FP_BUILTIN_NAMES = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sqrt",
    "exp",
    "log",
    "gamma",
    "cbrt",
    "floor",
    "ceil",
    "trunc",
]

UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES = ["isfinite", "isinf", "isnan"]


def _make_unary_math_builtin(name):
    def impl(value: common.Field | core_defs.ScalarT, /) -> common.Field | core_defs.ScalarT:
        # TODO(havogt): enable once we have a failing test (see `test_math_builtin_execution.py`)
        # assert core_defs.is_scalar_type(value) # default implementation for scalars, Fields are handled via dispatch # noqa: E800 # commented code
        # return getattr(math, name)(value)# noqa: E800 # commented code
        raise NotImplementedError()

    impl.__name__ = name
    globals()[name] = BuiltInFunction(impl)


for f in (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
):
    _make_unary_math_builtin(f)

BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]


def _make_binary_math_builtin(name):
    def impl(
        lhs: common.Field | core_defs.ScalarT,
        rhs: common.Field | core_defs.ScalarT,
        /,
    ) -> common.Field | core_defs.ScalarT:
        # default implementation for scalars, Fields are handled via dispatch
        assert core_defs.is_scalar_type(lhs)
        assert core_defs.is_scalar_type(rhs)
        return getattr(np, name)(lhs, rhs)

    impl.__name__ = name
    globals()[name] = BuiltInFunction(impl)


for f in BINARY_MATH_NUMBER_BUILTIN_NAMES:
    _make_binary_math_builtin(f)

MATH_BUILTIN_NAMES = (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + BINARY_MATH_NUMBER_BUILTIN_NAMES
)

FUN_BUILTIN_NAMES = [
    "neighbor_sum",
    "max_over",
    "min_over",
    "broadcast",
    "where",
    "astype",
    "as_offset",
] + MATH_BUILTIN_NAMES

BUILTIN_NAMES = TYPE_BUILTIN_NAMES + FUN_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in BUILTIN_NAMES}

__all__ = [*((set(BUILTIN_NAMES) | set(TYPE_ALIAS_NAMES)) - {"Dimension", "Field"})]


# TODO(tehrengruber): FieldOffset and runtime.Offset are not an exact conceptual
#  match. Revisit if we want to continue subclassing here. If we split
#  them also check whether Dimension should continue to be the shared or define
#  guidelines for decision.
@dataclasses.dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    source: common.Dimension
    target: tuple[common.Dimension] | tuple[common.Dimension, common.Dimension]

    @functools.cached_property
    def _cache(self) -> dict:
        return {}

    def __post_init__(self):
        if len(self.target) == 2 and self.target[1].kind != common.DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self):
        return ts.OffsetType(source=self.source, target=self.target)

    def __getitem__(self, offset: int) -> common.ConnectivityField:
        """Serve as a connectivity factory."""
        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.offset_provider.get(None)
        assert current_offset_provider is not None
        offset_definition = current_offset_provider[self.value]

        connectivity: common.ConnectivityField
        if isinstance(offset_definition, common.Dimension):
            connectivity = common.CartesianConnectivity(offset_definition, offset)
        elif isinstance(
            offset_definition, gtx.NeighborTableOffsetProvider
        ) or common.is_connectivity_field(offset_definition):
            unrestricted_connectivity = self.as_connectivity_field()
            assert unrestricted_connectivity.domain.ndim > 1
            named_index = (self.target[-1], offset)
            connectivity = unrestricted_connectivity[named_index]
        else:
            raise NotImplementedError()

        return connectivity

    def as_connectivity_field(self):
        """Convert to connectivity field using the offset providers in current embedded execution context."""
        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.offset_provider.get(None)
        assert current_offset_provider is not None
        offset_definition = current_offset_provider[self.value]

        cache_key = id(offset_definition)
        if (connectivity := self._cache.get(cache_key, None)) is None:
            if common.is_connectivity_field(offset_definition):
                connectivity = offset_definition
            elif isinstance(offset_definition, gtx.NeighborTableOffsetProvider):
                assert not offset_definition.has_skip_values
                connectivity = gtx.as_connectivity(
                    domain=self.target,
                    codomain=self.source,
                    data=offset_definition.table,
                    dtype=offset_definition.index_type,
                )
            else:
                raise NotImplementedError()

            self._cache[cache_key] = connectivity

        return connectivity
