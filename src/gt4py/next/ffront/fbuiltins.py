# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import inspect
import math
import operator
from builtins import bool, float, int, tuple  # noqa: A004 shadowing a Python built-in
from typing import Any, Callable, Final, Generic, ParamSpec, Tuple, TypeAlias, TypeVar, Union, cast

import numpy as np
from numpy import float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Dimension, Field  # noqa: F401 [unused-import] for TYPE_BUILTINS
from gt4py.next.iterator import runtime
from gt4py.next.type_system import type_specifications as ts


PYTHON_TYPE_BUILTINS = [bool, int, float, tuple]
PYTHON_TYPE_BUILTIN_NAMES = [t.__name__ for t in PYTHON_TYPE_BUILTINS]

TYPE_BUILTINS = [
    common.Field,
    common.Dimension,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
    *PYTHON_TYPE_BUILTINS,
]  # TODO(tehrengruber): validate matches iterator.builtins.TYPE_BUILTINS?

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
    elif t is FieldOffset:
        return ts.OffsetType
    elif t is common.Connectivity:
        return ts.OffsetType
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"{self.function.__module__}.{self.function.__name__}")
        object.__setattr__(self, "__doc__", self.function.__doc__)

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
            return tuple(self(mask, t, f) for t, f in zip(true_field, false_field))  # type: ignore[return-value] # `tuple` is not `_R`
        return super().__call__(mask, true_field, false_field)


@BuiltInFunction
def neighbor_sum(field: common.Field, /, axis: common.Dimension) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def max_over(field: common.Field, /, axis: common.Dimension) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def min_over(field: common.Field, /, axis: common.Dimension) -> common.Field:
    raise NotImplementedError()


@BuiltInFunction
def broadcast(
    field: common.Field | core_defs.ScalarT, dims: tuple[common.Dimension, ...], /
) -> common.Field:
    assert core_defs.is_scalar_type(
        field
    )  # default implementation for scalars, Fields are handled via dispatch
    # TODO(havogt) implement with FunctionField, the workaround is to ignore broadcasting on scalars as they broadcast automatically, but we lose the check for compatible dimensions
    return field  # type: ignore[return-value] # see comment above


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
    value: common.Field | core_defs.ScalarT | Tuple, type_: type, /
) -> common.Field | core_defs.ScalarT | Tuple:
    if isinstance(value, tuple):
        return tuple(astype(v, type_) for v in value)
    # default implementation for scalars, Fields are handled via dispatch
    assert core_defs.is_scalar_type(value)
    return core_defs.dtype(type_).scalar_type(value)


_UNARY_MATH_NUMBER_BUILTIN_IMPL: Final = {"abs": abs, "neg": operator.neg}
UNARY_MATH_NUMBER_BUILTIN_NAMES: Final = [*_UNARY_MATH_NUMBER_BUILTIN_IMPL.keys()]

_UNARY_MATH_FP_BUILTIN_IMPL: Final = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "arcsinh": math.asinh,
    "arccosh": math.acosh,
    "arctanh": math.atanh,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "gamma": math.gamma,
    "cbrt": math.cbrt if hasattr(math, "cbrt") else np.cbrt,  # match.cbrt() only added in 3.11
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
}
UNARY_MATH_FP_BUILTIN_NAMES: Final = [*_UNARY_MATH_FP_BUILTIN_IMPL.keys()]

_UNARY_MATH_FP_PREDICATE_BUILTIN_IMPL: Final = {
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
}
UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES: Final = [*_UNARY_MATH_FP_PREDICATE_BUILTIN_IMPL.keys()]


def _make_unary_math_builtin(name: str) -> None:
    _math_builtin = (
        _UNARY_MATH_NUMBER_BUILTIN_IMPL
        | _UNARY_MATH_FP_BUILTIN_IMPL
        | _UNARY_MATH_FP_PREDICATE_BUILTIN_IMPL
    )[name]

    def impl(value: common.Field | core_defs.ScalarT, /) -> common.Field | core_defs.ScalarT:
        # TODO(havogt): enable tests in `test_math_builtin_execution.py`
        assert core_defs.is_scalar_type(
            value
        )  # default implementation for scalars, Fields are handled via dispatch

        return cast(common.Field | core_defs.ScalarT, _math_builtin(value))  # type: ignore[operator] # calling a function of unknown type

    impl.__name__ = name
    globals()[name] = BuiltInFunction(impl)


for f in (
    UNARY_MATH_NUMBER_BUILTIN_NAMES
    + UNARY_MATH_FP_BUILTIN_NAMES
    + UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
):
    _make_unary_math_builtin(f)

BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]


def _make_binary_math_builtin(name: str) -> None:
    def impl(
        lhs: common.Field | core_defs.ScalarT, rhs: common.Field | core_defs.ScalarT, /
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
    *MATH_BUILTIN_NAMES,
]

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

    def __post_init__(self) -> None:
        if len(self.target) == 2 and self.target[1].kind != common.DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self) -> ts.OffsetType:
        return ts.OffsetType(source=self.source, target=self.target)

    def __getitem__(self, offset: int) -> common.Connectivity:
        """Serve as a connectivity factory."""
        from gt4py.next import embedded  # avoid circular import

        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.offset_provider.get(None)
        assert current_offset_provider is not None
        offset_definition = current_offset_provider[self.value]

        connectivity: common.Connectivity
        if isinstance(offset_definition, common.Dimension):
            connectivity = common.CartesianConnectivity(offset_definition, offset)
        elif isinstance(offset_definition, common.Connectivity):
            assert common.is_neighbor_connectivity(offset_definition)
            named_index = common.NamedIndex(self.target[-1], offset)
            connectivity = offset_definition[named_index]
        else:
            raise NotImplementedError()

        return connectivity

    def as_connectivity_field(self) -> common.Connectivity:
        """Convert to connectivity field using the offset providers in current embedded execution context."""
        from gt4py.next import embedded  # avoid circular import

        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.offset_provider.get(None)
        assert current_offset_provider is not None
        offset_definition = current_offset_provider[self.value]

        cache_key = id(offset_definition)
        if (connectivity := self._cache.get(cache_key, None)) is None:
            if isinstance(offset_definition, common.Connectivity):
                connectivity = offset_definition
            else:
                raise NotImplementedError()

            self._cache[cache_key] = connectivity

        return connectivity
