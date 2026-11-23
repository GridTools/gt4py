# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
from builtins import bool, float, int, tuple  # noqa: A004 shadowing a Python built-in
from types import UnionType
from typing import (
    Any,
    Callable,
    Concatenate,
    Final,
    Generic,
    NoReturn,
    ParamSpec,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from numpy import float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64

from gt4py._core import definitions as core_defs
from gt4py.next import common, named_collections
from gt4py.next.common import Dimension, Field  # noqa: F401 [unused-import] for TYPE_BUILTINS
from gt4py.next.iterator import runtime
from gt4py.next.type_system import type_specifications as ts


__all__ = [  # noqa: RUF022 [unsorted-dunder-all] # type: ignore[attr-defined]
    "bool",
    "float",
    "float32",
    "float64",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "tuple",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "IndexType",
    "neighbor_sum",
    "max_over",
    "min_over",
    "where",
    "broadcast",
    "astype",
    "abs",
    "neg",
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
    "minimum",
    "maximum",
    "fmod",
    "power",
    "isfinite",
    "isinf",
    "isnan",
]


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
    elif t is common.Domain:
        return ts.DomainType
    elif t is type:
        return (
            ts.ConstructorType
        )  # our type of type is currently represented by the type constructor function
    elif t is tuple or get_origin(t) is tuple:
        return ts.TupleType
    # 'Union[A, B]' and 'A | B' are different runtime objects: the latter is a
    # 'types.UnionType', which carries no '__origin__' at all.
    elif get_origin(t) in (Union, UnionType):
        member_types = [_type_conversion_helper(e) for e in get_args(t)]
        assert all(type(m) is type and issubclass(m, ts.TypeSpec) for m in member_types)
        # `cast` to break the recursion
        return cast(tuple[type[ts.TypeSpec], ...], tuple(member_types))
    elif t in named_collections.CUSTOM_NAMED_COLLECTION_TYPES:
        return ts.NamedCollectionType
    else:
        raise AssertionError("Illegal type encountered.")


def _type_conversion(t: type) -> ts.DeferredType:
    return ts.DeferredType(constraint=_type_conversion_helper(t))


@dataclasses.dataclass(frozen=True)
class BuiltInFunction(Generic[_R, _P]):
    """
    A builtin of the DSL, declared by `definition`: its signature types it, its docstring documents it.

    Calling it runs `call_implementation`, which `gt4py.next.embedded` sets.
    """

    name: str = dataclasses.field(init=False)
    definition: Callable[_P, _R]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"{self.definition.__module__}.{self.definition.__name__}")
        object.__setattr__(self, "__doc__", self.definition.__doc__)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return call_implementation(self, *args, **kwargs)

    def __gt_type__(self) -> ts.FunctionType:
        signature = inspect.signature(self.definition)
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


def _not_registered(builtin: BuiltInFunction, /, *args: Any, **kwargs: Any) -> NoReturn:
    raise RuntimeError(
        f"'{builtin.name}' was called with values, but no implementation of the builtins is"
        " registered: import 'gt4py.next.embedded'."
    )


# dependency inversion, as `iterator.runtime.fendef_embedded`: set by `gt4py.next.embedded` on import
call_implementation: Callable[Concatenate[BuiltInFunction, ...], Any] = _not_registered


CondT = TypeVar("CondT", bound=Union[common.Field, common.Domain])
FieldT1 = TypeVar(
    "FieldT1",
    bound=Union[common.Field, core_defs.Scalar, Tuple, named_collections.CustomNamedCollection],
)
FieldT2 = TypeVar(
    "FieldT2",
    bound=Union[common.Field, core_defs.Scalar, Tuple, named_collections.CustomNamedCollection],
)


class WhereBuiltinFunction(
    BuiltInFunction[_R, [CondT, FieldT1, FieldT2]],
    Generic[_R, CondT, FieldT1, FieldT2],
):
    @overload  # type: ignore[override] # this technically clashes with the superclass
    def __call__(  # type: ignore[overload-overlap] # without both overloads mypy raises false positives
        self,
        cond: CondT,
        true_field: common.Field | core_defs.ScalarT,
        false_field: common.Field | core_defs.ScalarT,
    ) -> common.Field: ...

    @overload
    def __call__(
        self,
        cond: CondT,
        true_field: Tuple | core_defs.ScalarT,
        false_field: Tuple | core_defs.ScalarT,
    ) -> Tuple: ...

    def __call__(self, cond: CondT, true_field: FieldT1, false_field: FieldT2) -> _R:  # type: ignore[misc] # supposedly this signature does not accept all the possible args allowed by the overloads ??
        return super().__call__(cond, true_field, false_field)


@BuiltInFunction
def neighbor_sum(field: common.Field, /, axis: common.Dimension) -> common.Field: ...  # type: ignore[empty-body]


@BuiltInFunction
def max_over(field: common.Field, /, axis: common.Dimension) -> common.Field: ...  # type: ignore[empty-body]


@BuiltInFunction
def min_over(field: common.Field, /, axis: common.Dimension) -> common.Field: ...  # type: ignore[empty-body]


@BuiltInFunction
def broadcast(  # type: ignore[empty-body]
    field: common.Field | core_defs.ScalarT, dims: tuple[common.Dimension, ...], /
) -> common.Field: ...


@WhereBuiltinFunction
def where(  # type: ignore[empty-body]
    mask: common.Field,
    true_field: common.Field | core_defs.ScalarT | Tuple | named_collections.CustomNamedCollection,
    false_field: common.Field | core_defs.ScalarT | Tuple | named_collections.CustomNamedCollection,
    /,
) -> common.Field | Tuple: ...


@overload
def astype(value: common.Field, type_: type, /) -> common.Field: ...


@overload
def astype(value: core_defs.ScalarT, type_: type, /) -> core_defs.ScalarT: ...


@overload
def astype(value: Tuple, type_: type, /) -> Tuple: ...


@BuiltInFunction
def astype(  # type: ignore[empty-body]
    value: common.Field | core_defs.ScalarT | Tuple, type_: type, /
) -> common.Field | core_defs.ScalarT | Tuple: ...


UNARY_MATH_NUMBER_BUILTIN_NAMES: Final = ["abs", "neg"]
UNARY_MATH_FP_BUILTIN_NAMES: Final = [
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
UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES: Final = ["isfinite", "isinf", "isnan"]


def _make_unary_math_builtin(name: str) -> BuiltInFunction:
    def definition(  # type: ignore[empty-body]
        value: common.Field | core_defs.ScalarT, /
    ) -> common.Field | core_defs.ScalarT: ...

    definition.__name__ = name
    return BuiltInFunction(definition)


abs = _make_unary_math_builtin("abs")  # noqa: A001 [shadowing]
neg = _make_unary_math_builtin("neg")
sin = _make_unary_math_builtin("sin")
cos = _make_unary_math_builtin("cos")
tan = _make_unary_math_builtin("tan")
arcsin = _make_unary_math_builtin("arcsin")
arccos = _make_unary_math_builtin("arccos")
arctan = _make_unary_math_builtin("arctan")
sinh = _make_unary_math_builtin("sinh")
cosh = _make_unary_math_builtin("cosh")
tanh = _make_unary_math_builtin("tanh")
arcsinh = _make_unary_math_builtin("arcsinh")
arccosh = _make_unary_math_builtin("arccosh")
arctanh = _make_unary_math_builtin("arctanh")
sqrt = _make_unary_math_builtin("sqrt")
exp = _make_unary_math_builtin("exp")
log = _make_unary_math_builtin("log")
gamma = _make_unary_math_builtin("gamma")
cbrt = _make_unary_math_builtin("cbrt")
floor = _make_unary_math_builtin("floor")
ceil = _make_unary_math_builtin("ceil")
trunc = _make_unary_math_builtin("trunc")
isfinite = _make_unary_math_builtin("isfinite")
isinf = _make_unary_math_builtin("isinf")
isnan = _make_unary_math_builtin("isnan")


BINARY_MATH_NUMBER_BUILTIN_NAMES = ["minimum", "maximum", "fmod", "power"]


def _make_binary_math_builtin(name: str) -> BuiltInFunction:
    def definition(  # type: ignore[empty-body]
        lhs: common.Field | core_defs.ScalarT, rhs: common.Field | core_defs.ScalarT, /
    ) -> common.Field | core_defs.ScalarT: ...

    definition.__name__ = name
    return BuiltInFunction(definition)


minimum = _make_binary_math_builtin("minimum")
maximum = _make_binary_math_builtin("maximum")
fmod = _make_binary_math_builtin("fmod")
power = _make_binary_math_builtin("power")


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

assert all(isinstance(globals()[f], BuiltInFunction) for f in FUN_BUILTIN_NAMES), (
    "Missing builtin function"
)

BUILTIN_NAMES = TYPE_BUILTIN_NAMES + FUN_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in BUILTIN_NAMES}

should_export = (set(BUILTIN_NAMES) | set(TYPE_ALIAS_NAMES)) - {"Dimension", "Field"}
actual_export = set(__all__)
assert (diff := should_export - actual_export) == set(), (
    f"Missing symbol(s) in 'fbuiltins.__all__': {diff}"
)
assert (diff := actual_export - should_export) == set(), (
    f"Symbol(s) exported but not defined in 'fbuiltins': {diff}"
)


# TODO(tehrengruber): FieldOffset and runtime.Offset are not an exact conceptual
#  match. Revisit if we want to continue subclassing here. If we split
#  them also check whether Dimension should continue to be the shared or define
#  guidelines for decision.
@dataclasses.dataclass(frozen=True)
class FieldOffset(runtime.Offset):
    #: The tag, i.e. the offset-provider key.
    value: str
    source: common.Dimension
    target: tuple[common.Dimension] | tuple[common.Dimension, common.Dimension]

    def __post_init__(self) -> None:
        if len(self.target) == 2 and self.target[1].kind != common.DimensionKind.LOCAL:
            raise ValueError("Second dimension in offset must be a local dimension.")

    def __gt_type__(self) -> ts.OffsetType:
        return ts.OffsetType(source=self.source, target=self.target, tag=self.value)

    def __getitem__(self, offset: int) -> common.Connectivity:
        """Serve as a connectivity factory."""
        from gt4py.next import embedded  # avoid circular import

        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.get_offset_provider(None)
        assert current_offset_provider is not None
        offset_definition = common.get_offset(current_offset_provider, self.value)

        assert common.is_neighbor_table(offset_definition)
        named_index = common.NamedIndex(self.target[-1], offset)
        connectivity = offset_definition[named_index]

        return connectivity

    def as_connectivity_field(self) -> common.Connectivity:
        """Convert to connectivity field using the offset providers in current embedded execution context."""
        from gt4py.next import embedded  # avoid circular import

        assert isinstance(self.value, str)
        current_offset_provider = embedded.context.get_offset_provider(None)
        assert current_offset_provider is not None
        offset_definition = common.get_offset(current_offset_provider, self.value)

        if not isinstance(offset_definition, common.Connectivity):
            raise NotImplementedError()

        return offset_definition


def is_cartesian_offset(offset: FieldOffset | ts.OffsetType) -> bool:
    return (
        len(offset.target) == 1
        and offset.source == offset.target[0]
        and offset.source.kind == offset.target[0].kind
        and offset.target[0].kind != common.DimensionKind.LOCAL
    )
