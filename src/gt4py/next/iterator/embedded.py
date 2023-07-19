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

# TODO(havogt) move public definitions and make this module private

from __future__ import annotations

import abc
import contextvars as cvars
import copy
import dataclasses
import itertools
import math
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    SupportsFloat,
    SupportsInt,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.iterator import builtins, runtime
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_specifications as ts, type_translation


EMBEDDED = "embedded"


# Atoms
Tag: TypeAlias = str
IntIndex: TypeAlias = int | np.integer

ArrayIndex: TypeAlias = slice | IntIndex
ArrayIndexOrIndices: TypeAlias = ArrayIndex | tuple[ArrayIndex, ...]

FieldIndex: TypeAlias = (
    range | slice | IntIndex
)  # A `range` FieldIndex can be negative indicating a relative position with respect to origin, not wrap-around semantics like `slice` TODO(havogt): remove slice here
FieldIndices: TypeAlias = tuple[FieldIndex, ...]
FieldIndexOrIndices: TypeAlias = FieldIndex | FieldIndices

FieldAxis: TypeAlias = (
    common.Dimension | runtime.Offset
)  # TODO Offset should be removed, is sometimes used for sparse dimensions
TupleAxis: TypeAlias = type[None]
Axis: TypeAlias = Union[FieldAxis, TupleAxis]
Scalar: TypeAlias = (
    SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64 | np.bool_
)


class SparseTag(Tag):
    ...


class NeighborTableOffsetProvider:
    def __init__(
        self,
        table: npt.NDArray,
        origin_axis: common.Dimension,
        neighbor_axis: common.Dimension,
        max_neighbors: int,
        has_skip_values=True,
    ) -> None:
        self.table = table
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        assert not hasattr(table, "shape") or table.shape[1] == max_neighbors
        self.max_neighbors = max_neighbors
        self.has_skip_values = has_skip_values
        self.index_type = table.dtype

    def mapped_index(self, primary: IntIndex, neighbor_idx: IntIndex) -> IntIndex:
        return self.table[(primary, neighbor_idx)]

    def __gtfn_bindings__(self, name: str):
        GENERATED_CONNECTIVITY_PARAM_PREFIX = "gt_conn_"
        param = interface.Parameter(
            name=GENERATED_CONNECTIVITY_PARAM_PREFIX + name.lower(),
            type_=ts.FieldType(
                dims=[self.origin_axis, common.Dimension(name)],
                dtype=ts.ScalarType(type_translation.get_scalar_kind(self.index_type)),
            ),
        )
        nbtbl = (
            f"gridtools::fn::sid_neighbor_table::as_neighbor_table<"
            f"generated::{self.origin_axis.value}_t, "
            f"generated::{name}_t, {self.max_neighbors}"
            f">(std::forward<decltype({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()})>({GENERATED_CONNECTIVITY_PARAM_PREFIX}{name.lower()}))"
        )
        return param, nbtbl, self.table  # TODO table probably goes together with param


class StridedNeighborOffsetProvider:
    def __init__(
        self,
        origin_axis: common.Dimension,
        neighbor_axis: common.Dimension,
        max_neighbors: int,
        has_skip_values=True,
    ) -> None:
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        self.max_neighbors = max_neighbors
        self.has_skip_values = has_skip_values
        self.index_type = int

    def mapped_index(self, primary: IntIndex, neighbor_idx: IntIndex) -> IntIndex:
        return primary * self.max_neighbors + neighbor_idx

    def __gtfn_bindings__(self, _: str):
        elems = (f"i*{self.max_neighbors}+{n}" for n in range(self.max_neighbors))
        return None, f"[](auto i){{return std::tuple({','.join(elems)});}}", None


# Offsets
OffsetPart: TypeAlias = Tag | IntIndex
CompleteOffset: TypeAlias = tuple[Tag, IntIndex]
OffsetProviderElem: TypeAlias = common.Dimension | common.Connectivity
OffsetProvider: TypeAlias = dict[Tag, OffsetProviderElem]

# Positions
SparsePositionEntry = list[int]
IncompleteSparsePositionEntry: TypeAlias = list[Optional[int]]
PositionEntry: TypeAlias = SparsePositionEntry | IntIndex
IncompletePositionEntry: TypeAlias = IncompleteSparsePositionEntry | IntIndex
ConcretePosition: TypeAlias = dict[Tag, PositionEntry]
IncompletePosition: TypeAlias = dict[Tag, IncompletePositionEntry]

Position: TypeAlias = Union[ConcretePosition, IncompletePosition]
#: A ``None`` position flags invalid not-a-neighbor results in neighbor-table lookups
MaybePosition: TypeAlias = Optional[Position]


def is_int_index(p: Any) -> TypeGuard[IntIndex]:
    return isinstance(p, (int, np.integer))


def _tupelize(tup):
    if isinstance(tup, tuple):
        return tup
    else:
        return (tup,)


@runtime_checkable
class ItIterator(Protocol):
    """
    Prototype for the Iterator concept of Iterator IR.

    `ItIterator` to avoid name clashes with `Iterator` from `typing` and `collections.abc`.
    """

    def shift(self, *offsets: OffsetPart) -> ItIterator:
        ...

    def can_deref(self) -> bool:
        ...

    def deref(self) -> Any:
        ...


@runtime_checkable
class LocatedField(Protocol):
    """A field with named dimensions providing read access."""

    @property
    @abc.abstractmethod
    def axes(self) -> tuple[common.Dimension, ...]:
        ...

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        ...

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple([0] * len(self.axes))


class MutableLocatedField(LocatedField, Protocol):
    """A LocatedField with write access."""

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_setitem(self, indices: FieldIndexOrIndices, value: Any) -> None:
        ...


#: Column range used in column mode (`column_axis != None`) in the current closure execution context.
column_range_cvar: cvars.ContextVar[range] = cvars.ContextVar("column_range")
#: Offset provider dict in the current closure execution context.
offset_provider_cvar: cvars.ContextVar[OffsetProvider] = cvars.ContextVar("offset_provider")


class Column(np.lib.mixins.NDArrayOperatorsMixin):
    """Represents a column when executed in column mode (`column_axis != None`).

    Implements `__array_ufunc__` and `__array_function__` to isolate
    and simplify dispatching in iterator ir builtins.
    """

    def __init__(self, kstart: int, data: np.ndarray | Scalar) -> None:
        self.kstart = kstart
        assert isinstance(data, (np.ndarray, Scalar))  # type: ignore # mypy bug #11673
        column_range = column_range_cvar.get()
        self.data = data if isinstance(data, np.ndarray) else np.full(len(column_range), data)

    def __getitem__(self, i: int) -> Any:
        result = self.data[i - self.kstart]
        #  numpy type
        if self.data.dtype.names:
            return tuple(result)
        return result

    def tuple_get(self, i: int) -> Column:
        if self.data.dtype.names:
            return Column(self.kstart, self.data[self.data.dtype.names[i]])
        else:
            return Column(self.kstart, self.data[i, ...])

    def __setitem__(self, i: int, v: Any) -> None:
        self.data[i - self.kstart] = v

    def __array__(self, dtype: Optional[npt.DTypeLike] = None) -> np.ndarray:
        return self.data.astype(dtype, copy=False)

    def _validate_kstart(self, args):
        if wrong_kstarts := (  # noqa: F841 # wrong_kstarts looks unused
            set(arg.kstart for arg in args if isinstance(arg, Column)) - {self.kstart}
        ):
            raise ValueError(
                "Incompatible Column.kstart: it should be '{self.kstart}' but found other values: {wrong_kstarts}"
            )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Column:
        # note:
        # - we allow scalars to silently pass through and be handled correctly by numpy
        # - we let numpy do the checking of compatible shapes
        assert method == "__call__"
        self._validate_kstart(inputs)
        return self.__class__(
            self.kstart,
            ufunc(*(inp.data if isinstance(inp, Column) else inp for inp in inputs), **kwargs),
        )

    def __array_function__(self, func, types, args, kwargs) -> Column:
        # see note in `__array_ufunc__`
        self._validate_kstart(args)
        return self.__class__(
            self.kstart,
            func(*(arg.data if isinstance(arg, Column) else arg for arg in args), **kwargs),
        )


@builtins.deref.register(EMBEDDED)
def deref(it):
    return it.deref()


@builtins.can_deref.register(EMBEDDED)
def can_deref(it):
    return it.can_deref()


@builtins.if_.register(EMBEDDED)
def if_(cond, t, f):
    # ensure someone doesn't accidentally pass an iterator
    assert not hasattr(cond, "shift")
    if any(isinstance(arg, Column) for arg in (cond, t, f)):
        return np.where(cond, t, f)
    return t if cond else f


@builtins.cast_.register(EMBEDDED)
def cast_(obj, new_dtype):
    if isinstance(obj, Column):
        return obj.data.astype(new_dtype.__name__)
    return new_dtype(obj)


@builtins.not_.register(EMBEDDED)
def not_(a):
    if isinstance(a, Column):
        return np.logical_not(a.data)
    return not a


@builtins.and_.register(EMBEDDED)
def and_(a, b):
    if isinstance(a, Column):
        return np.logical_and(a, b)
    return a and b


@builtins.or_.register(EMBEDDED)
def or_(a, b):
    if isinstance(a, Column):
        return np.logical_or(a, b)
    return a or b


@builtins.xor_.register(EMBEDDED)
def xor_(a, b):
    if isinstance(a, Column):
        return np.logical_xor(a, b)
    return a ^ b


@builtins.tuple_get.register(EMBEDDED)
def tuple_get(i, tup):
    if isinstance(tup, Column):
        return tup.tuple_get(i)
    return tup[i]


@builtins.make_tuple.register(EMBEDDED)
def make_tuple(*args):
    return (*args,)


@builtins.lift.register(EMBEDDED)
def lift(stencil):
    def impl(*args):
        class _WrappedIterator:
            def __init__(
                self, stencil, args, *, offsets: Optional[list[OffsetPart]] = None, elem=None
            ) -> None:
                assert not offsets or all(isinstance(o, (int, str)) for o in offsets)
                self.stencil = stencil
                self.args = args
                self.offsets = offsets or []
                self.elem = elem

            # TODO needs to be supported by all iterators that represent tuples
            def __getitem__(self, index):
                return _WrappedIterator(self.stencil, self.args, offsets=self.offsets, elem=index)

            def shift(self, *offsets: OffsetPart):
                return _WrappedIterator(
                    self.stencil, self.args, offsets=[*self.offsets, *offsets], elem=self.elem
                )

            def _shifted_args(self):
                return tuple(map(lambda arg: arg.shift(*self.offsets), self.args))

            def can_deref(self):
                shifted_args = self._shifted_args()
                return all(shifted_arg.can_deref() for shifted_arg in shifted_args)

            def deref(self):
                if not self.can_deref():
                    # this can legally happen in cases like `if_(can_deref(lifted), deref(lifted), 42.)`
                    # because both branches will be eagerly executed
                    return _UNDEFINED

                shifted_args = self._shifted_args()

                if self.elem is None:
                    return self.stencil(*shifted_args)
                else:
                    return self.stencil(*shifted_args)[self.elem]

        return _WrappedIterator(stencil, args)

    return impl


NamedRange: TypeAlias = tuple[Tag | common.Dimension, range]


@builtins.cartesian_domain.register(EMBEDDED)
def cartesian_domain(*args: NamedRange) -> runtime.CartesianDomain:
    return runtime.CartesianDomain(args)


@builtins.unstructured_domain.register(EMBEDDED)
def unstructured_domain(*args: NamedRange) -> runtime.UnstructuredDomain:
    return runtime.UnstructuredDomain(args)


Domain: TypeAlias = (
    runtime.CartesianDomain | runtime.UnstructuredDomain | dict[str | common.Dimension, range]
)


@builtins.named_range.register(EMBEDDED)
def named_range(tag: Tag | common.Dimension, start: int, end: int) -> NamedRange:
    # TODO revisit this pattern after the discussion of 0d-field vs scalar
    if isinstance(start, ConstantField):
        start = start.value
    if isinstance(end, ConstantField):
        end = end.value
    return (tag, range(start, end))


@builtins.minus.register(EMBEDDED)
def minus(first, second):
    return first - second


@builtins.plus.register(EMBEDDED)
def plus(first, second):
    return first + second


@builtins.multiplies.register(EMBEDDED)
def multiplies(first, second):
    return first * second


@builtins.divides.register(EMBEDDED)
def divides(first, second):
    return first / second


@builtins.floordiv.register(EMBEDDED)
def floordiv(first, second):
    return first // second


@builtins.mod.register(EMBEDDED)
def mod(first, second):
    return first % second


@builtins.eq.register(EMBEDDED)
def eq(first, second):
    return first == second


@builtins.greater.register(EMBEDDED)
def greater(first, second):
    return first > second


@builtins.less.register(EMBEDDED)
def less(first, second):
    return first < second


@builtins.less_equal.register(EMBEDDED)
def less_equal(first, second):
    return first <= second


@builtins.greater_equal.register(EMBEDDED)
def greater_equal(first, second):
    return first >= second


@builtins.not_eq.register(EMBEDDED)
def not_eq(first, second):
    return first != second


CompositeOfScalarOrField: TypeAlias = Scalar | LocatedField | tuple["CompositeOfScalarOrField", ...]


def is_dtype_like(t: Any) -> TypeGuard[npt.DTypeLike]:
    return issubclass(t, (np.generic, int, float))


def infer_dtype_like_type(t: Any) -> npt.DTypeLike:
    res = xtyping.infer_type(t)
    assert is_dtype_like(res), res
    return res


def promote_scalars(val: CompositeOfScalarOrField):
    """Given a scalar, field or composite thereof promote all (contained) scalars to fields."""
    if isinstance(val, tuple):
        return tuple(promote_scalars(el) for el in val)
    elif isinstance(val, LocatedField):
        return val
    val_type = infer_dtype_like_type(val)
    if isinstance(val, Scalar):  # type: ignore # mypy bug
        return constant_field(val)
    else:
        raise ValueError(
            f"Expected a `Field` or a number (`float`, `np.int64`, ...), but got {val_type}."
        )


for math_builtin_name in builtins.MATH_BUILTINS:
    python_builtins = {"int": int, "float": float, "bool": bool, "str": str}
    decorator = getattr(builtins, math_builtin_name).register(EMBEDDED)
    impl: Callable
    if math_builtin_name == "gamma":
        # numpy has no gamma function
        impl = np.vectorize(math.gamma)
    elif math_builtin_name in python_builtins:
        # TODO: Should potentially use numpy fixed size types to be consistent
        #   with compiled backends. Currently using Python types to preserve
        #   existing behaviour.
        impl = python_builtins[math_builtin_name]
    else:
        impl = getattr(np, math_builtin_name)
    globals()[math_builtin_name] = decorator(impl)


def _lookup_offset_provider(offset_provider: OffsetProvider, tag: Tag) -> OffsetProviderElem:
    if tag not in offset_provider:
        raise RuntimeError(f"Missing offset provider for `{tag}`")
    return offset_provider[tag]


def _get_connectivity(offset_provider: OffsetProvider, tag: Tag) -> common.Connectivity:
    if not isinstance(
        connectivity := _lookup_offset_provider(offset_provider, tag), common.Connectivity
    ):
        raise RuntimeError(f"Expected a `Connectivity` for `{tag}`")
    return connectivity


def _named_range(axis: str, range_: Iterable[int]) -> Iterable[CompleteOffset]:
    return ((axis, i) for i in range_)


def _domain_iterator(domain: dict[Tag, range]) -> Iterable[Position]:
    return (
        dict(elem)
        for elem in itertools.product(*(_named_range(axis, rang) for axis, rang in domain.items()))
    )


def execute_shift(
    pos: Position, tag: Tag, index: IntIndex, *, offset_provider: OffsetProvider
) -> MaybePosition:
    assert pos is not None
    if isinstance(tag, SparseTag):
        current_entry = pos[tag]
        assert isinstance(current_entry, list)
        new_entry = list(current_entry)
        assert None in new_entry
        assert isinstance(
            index, int
        )  # narrowing to `int` as it's an element of `SparsePositionEntry`
        for i, p in reversed(list(enumerate(new_entry))):
            # first shift applies to the last sparse dimensions of that axis type
            if p is None:
                new_entry[i] = index
                break
        # the assertions above confirm pos is incomplete casting here to avoid duplicating work in a type guard
        return cast(IncompletePosition, pos) | {tag: new_entry}

    assert tag in offset_provider
    offset_implementation = offset_provider[tag]
    if isinstance(offset_implementation, common.Dimension):
        new_pos = copy.copy(pos)
        if is_int_index(value := new_pos[offset_implementation.value]):
            new_pos[offset_implementation.value] = value + index
        else:
            raise AssertionError()
        return new_pos
    else:
        assert isinstance(offset_implementation, common.Connectivity)
        assert offset_implementation.origin_axis.value in pos
        new_pos = pos.copy()
        new_pos.pop(offset_implementation.origin_axis.value)
        cur_index = pos[offset_implementation.origin_axis.value]
        assert is_int_index(cur_index)
        if offset_implementation.mapped_index(cur_index, index) in [
            None,
            -1,
        ]:
            return None
        else:
            new_index = offset_implementation.mapped_index(cur_index, index)
            assert new_index is not None
            new_pos[offset_implementation.neighbor_axis.value] = int(new_index)

        return new_pos

    raise AssertionError("Unknown object in `offset_provider`")


def _is_list_of_complete_offsets(
    complete_offsets: list[tuple[Any, Any]]
) -> TypeGuard[list[CompleteOffset]]:
    return all(
        isinstance(tag, Tag) and isinstance(offset, (int, np.integer))
        for tag, offset in complete_offsets
    )


def group_offsets(*offsets: OffsetPart) -> list[CompleteOffset]:
    assert len(offsets) % 2 == 0
    complete_offsets = [*zip(offsets[::2], offsets[1::2])]
    assert _is_list_of_complete_offsets(
        complete_offsets
    ), f"Invalid sequence of offset parts: {offsets}"
    return complete_offsets


def shift_position(
    pos: MaybePosition, *complete_offsets: CompleteOffset, offset_provider: OffsetProvider
) -> MaybePosition:
    if pos is None:
        return None
    new_pos = pos.copy()
    for tag, index in complete_offsets:
        if (
            shifted_pos := execute_shift(new_pos, tag, index, offset_provider=offset_provider)
        ) is not None:
            new_pos = shifted_pos
        else:
            return None
    return new_pos


class Undefined:
    def __float__(self):
        return np.nan

    @classmethod
    def _setup_math_operations(cls):
        ops = [
            "__add__",
            "__sub__",
            "__mul__",
            "__matmul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__divmod__",
            "__pow__",
            "__lshift__",
            "__rshift__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__eq__",
            "__ne__",
            "__and__",
            "__xor__",
            "__or__",
            "__radd__",
            "__rsub__",
            "__rmul__",
            "__rmatmul__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rmod__",
            "__rdivmod__",
            "__rpow__",
            "__rlshift__",
            "__rrshift__",
            "__rand__",
            "__rxor__",
            "__ror__",
            "__neg__",
            "__pos__",
            "__abs__",
            "__invert__",
        ]
        for op in ops:
            setattr(cls, op, lambda self, *args, **kwargs: _UNDEFINED)


Undefined._setup_math_operations()

_UNDEFINED = Undefined()


def _is_concrete_position(pos: Position) -> TypeGuard[ConcretePosition]:
    return all(
        isinstance(v, (int, np.integer))
        or (isinstance(v, list) and all(isinstance(e, (int, np.integer)) for e in v))
        for v in pos.values()
    )


def _get_axes(
    field_or_tuple: LocatedField | tuple,
) -> Sequence[common.Dimension | runtime.Offset]:  # arbitrary nesting of tuples of LocatedField
    return (
        _get_axes(field_or_tuple[0]) if isinstance(field_or_tuple, tuple) else field_or_tuple.axes
    )


def _single_vertical_idx(
    indices: FieldIndices, column_axis_idx: int, column_index: IntIndex
) -> tuple[FieldIndex, ...]:
    transformed = tuple(
        index if i != column_axis_idx else column_index for i, index in enumerate(indices)
    )
    return transformed


@overload
def _make_tuple(
    field_or_tuple: tuple[tuple | LocatedField, ...],  # arbitrary nesting of tuples of LocatedField
    indices: FieldIndices,
    *,
    column_axis: Tag,
) -> tuple[tuple | Column, ...]:
    ...


@overload
def _make_tuple(
    field_or_tuple: tuple[tuple | LocatedField, ...],  # arbitrary nesting of tuples of LocatedField
    indices: FieldIndices,
    *,
    column_axis: Literal[None] = None,
) -> tuple[tuple | npt.DTypeLike, ...]:  # arbitrary nesting
    ...


@overload
def _make_tuple(field_or_tuple: LocatedField, indices: FieldIndices, *, column_axis: Tag) -> Column:
    ...


@overload
def _make_tuple(
    field_or_tuple: LocatedField, indices: FieldIndices, *, column_axis: Literal[None] = None
) -> npt.DTypeLike:
    ...


def _make_tuple(
    field_or_tuple: LocatedField | tuple[tuple | LocatedField, ...],
    indices: FieldIndices,
    *,
    column_axis: Optional[Tag] = None,
) -> Column | npt.DTypeLike | tuple[tuple | Column | npt.DTypeLike, ...]:
    column_range = column_range_cvar.get()
    if isinstance(field_or_tuple, tuple):
        if column_axis is not None:
            assert column_range
            # construct a Column of tuples
            column_axis_idx = _axis_idx(_get_axes(field_or_tuple), column_axis)
            if column_axis_idx is None:
                column_axis_idx = -1  # field doesn't have the column index, e.g. ContantField
            first = tuple(
                _make_tuple(f, _single_vertical_idx(indices, column_axis_idx, column_range.start))
                for f in field_or_tuple
            )
            col = Column(
                column_range.start, np.zeros(len(column_range), dtype=_column_dtype(first))
            )
            col[0] = first
            for i in column_range[1:]:
                col[i] = tuple(
                    _make_tuple(f, _single_vertical_idx(indices, column_axis_idx, i))
                    for f in field_or_tuple
                )
            return col
        else:
            return tuple(_make_tuple(f, indices) for f in field_or_tuple)
    else:
        data = field_or_tuple.field_getitem(indices)
        if column_axis is not None:
            # wraps a vertical slice of an input field into a `Column`
            assert column_range is not None
            return Column(column_range.start, data)
        else:
            return data


def _axis_idx(axes: Sequence[common.Dimension | runtime.Offset], axis: Tag) -> Optional[int]:
    for i, a in enumerate(axes):
        if a.value == axis:
            return i
    return None


@dataclasses.dataclass(frozen=True)
class MDIterator:
    field: LocatedField
    pos: MaybePosition
    column_axis: Optional[Tag] = dataclasses.field(default=None, kw_only=True)

    def shift(self, *offsets: OffsetPart) -> MDIterator:
        complete_offsets = group_offsets(*offsets)
        offset_provider = offset_provider_cvar.get()
        assert offset_provider is not None
        return MDIterator(
            self.field,
            shift_position(self.pos, *complete_offsets, offset_provider=offset_provider),
            column_axis=self.column_axis,
        )

    def can_deref(self) -> bool:
        return self.pos is not None

    def deref(self) -> Any:
        if not self.can_deref():
            # this can legally happen in cases like `if_(can_deref(inp), deref(inp), 42.)`
            # because both branches will be eagerly executed
            return _UNDEFINED

        assert self.pos is not None
        shifted_pos = self.pos.copy()
        axes = _get_axes(self.field)

        if __debug__:
            if not all(axis.value in shifted_pos.keys() for axis in axes if axis is not None):
                raise IndexError("Iterator position doesn't point to valid location for its field.")
        slice_column = dict[Tag, range]()
        column_range = column_range_cvar.get()
        if self.column_axis is not None:
            assert column_range is not None
            k_pos = shifted_pos.pop(self.column_axis)
            assert isinstance(k_pos, int)
            # the following range describes a range in the field
            # (negative values are relative to the origin, not relative to the size)
            slice_column[self.column_axis] = range(k_pos, k_pos + len(column_range))

        assert _is_concrete_position(shifted_pos)
        ordered_indices = get_ordered_indices(
            axes,
            {**shifted_pos, **slice_column},
        )
        return _make_tuple(
            self.field,
            ordered_indices,
            column_axis=self.column_axis,
        )


def _get_sparse_dimensions(axes: Sequence[common.Dimension | runtime.Offset]) -> list[Tag]:
    return [
        cast(Tag, axis.value)  # axis.value is always `str`
        for axis in axes
        if isinstance(axis, runtime.Offset)
        or (isinstance(axis, common.Dimension) and axis.kind == common.DimensionKind.LOCAL)
    ]


def make_in_iterator(
    inp: LocatedField,
    pos: Position,
    *,
    column_axis: Optional[Tag],
) -> ItIterator:
    axes = _get_axes(inp)
    sparse_dimensions = _get_sparse_dimensions(axes)
    new_pos: Position = pos.copy()
    for sparse_dim in set(sparse_dimensions):
        init = [None] * sparse_dimensions.count(sparse_dim)
        new_pos[sparse_dim] = init  # type: ignore[assignment] # looks like mypy is confused
    if column_axis is not None:
        column_range = column_range_cvar.get()
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        assert column_range is not None
        new_pos[column_axis] = column_range.start
    it = MDIterator(
        inp,
        new_pos,
        column_axis=column_axis,
    )
    if len(sparse_dimensions) >= 1:
        if len(sparse_dimensions) == 1:
            return SparseListIterator(it, sparse_dimensions[0])
        else:
            raise NotImplementedError(
                f"More than one local dimension is currently not supported, got {sparse_dimensions}"
            )
    else:
        return it


builtins.builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


class LocatedFieldImpl(MutableLocatedField):
    """A Field with named dimensions/axes."""

    @property
    def axes(self) -> tuple[common.Dimension, ...]:
        return self._axes

    def __init__(
        self,
        getter: Callable[[FieldIndexOrIndices], Any],
        axes: tuple[common.Dimension, ...],
        dtype,
        *,
        setter: Callable[[FieldIndexOrIndices, Any], None],
        array: Callable[[], npt.NDArray],
        origin: Optional[dict[common.Dimension, int]] = None,
    ):
        self.getter = getter
        self._axes = axes
        self.setter = setter
        self.array = array
        self.dtype = dtype
        self.origin = origin

    def __getitem__(self, indices: ArrayIndexOrIndices) -> Any:
        return self.array()[indices]

    # TODO in a stable implementation of the Field concept we should make this behavior the default behavior for __getitem__
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        indices = _tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices: ArrayIndexOrIndices, value: Any):
        self.array()[indices] = value

    def field_setitem(self, indices: FieldIndexOrIndices, value: Any):
        self.setter(indices, value)

    def __array__(self) -> np.ndarray:
        return self.array()

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        if not self.origin:
            return tuple([0] * len(self.axes))
        return cast(
            tuple[int], get_ordered_indices(self.axes, {k.value: v for k, v in self.origin.items()})
        )

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array().shape


def _is_field_axis(axis: Axis) -> TypeGuard[FieldAxis]:
    return isinstance(axis, FieldAxis)  # type: ignore[misc,arg-type] # see https://github.com/python/mypy/issues/11673


def _is_tuple_axis(axis: Axis) -> TypeGuard[TupleAxis]:
    return axis is None


def _is_sparse_position_entry(
    pos: FieldIndex | SparsePositionEntry,
) -> TypeGuard[SparsePositionEntry]:
    return isinstance(pos, list)


def get_ordered_indices(
    axes: Iterable[Axis], pos: Mapping[Tag, FieldIndex | SparsePositionEntry]
) -> tuple[FieldIndex, ...]:
    res: list[FieldIndex] = []
    sparse_position_tracker: dict[Tag, int] = {}
    for axis in axes:
        if _is_tuple_axis(axis):
            res.append(slice(None))
        else:
            assert _is_field_axis(axis)
            assert axis.value in pos
            assert isinstance(axis.value, str)
            elem = pos[axis.value]
            if _is_sparse_position_entry(elem):
                sparse_position_tracker.setdefault(axis.value, 0)
                res.append(elem[sparse_position_tracker[axis.value]])
                sparse_position_tracker[axis.value] += 1
            else:
                assert isinstance(elem, (int, np.integer, slice, range))
                res.append(elem)
    return tuple(res)


@overload
def _shift_range(range_or_index: range, offset: int) -> slice:
    ...


@overload
def _shift_range(range_or_index: IntIndex, offset: int) -> IntIndex:
    ...


def _shift_range(range_or_index: range | IntIndex, offset: int) -> ArrayIndex:
    if isinstance(range_or_index, range):
        # range_or_index describes a range in the field
        assert range_or_index.step == 1
        return slice(range_or_index.start + offset, range_or_index.stop + offset)
    else:
        assert is_int_index(range_or_index)
        return range_or_index + offset


@overload
def _range2slice(r: range) -> slice:
    ...


@overload
def _range2slice(r: IntIndex) -> IntIndex:
    ...


def _range2slice(r: range | IntIndex) -> slice | IntIndex:
    if isinstance(r, range):
        assert r.start >= 0 and r.stop >= r.start
        return slice(r.start, r.stop)
    return r


def _shift_field_indices(
    ranges_or_indices: tuple[range | IntIndex, ...],
    offsets: tuple[int, ...],
) -> tuple[ArrayIndex, ...]:
    return tuple(
        _range2slice(r) if o == 0 else _shift_range(r, o)
        for r, o in zip(ranges_or_indices, offsets)
    )


def np_as_located_field(
    *axes: common.Dimension, origin: Optional[dict[common.Dimension, int]] = None
) -> Callable[[np.ndarray], LocatedFieldImpl]:
    def _maker(a: np.ndarray) -> LocatedFieldImpl:
        if a.ndim != len(axes):
            raise TypeError("ndarray.ndim incompatible with number of given axes")

        if origin is not None:
            offsets = get_ordered_indices(axes, {k.value: v for k, v in origin.items()})
        else:
            offsets = None

        def setter(indices, value):
            indices = _tupelize(indices)
            a[_shift_field_indices(indices, offsets) if offsets else indices] = value

        def getter(indices):
            return a[_shift_field_indices(indices, offsets) if offsets else indices]

        return LocatedFieldImpl(
            getter,
            axes,
            dtype=a.dtype,
            setter=setter,
            array=a.__array__,
            origin=origin,
        )

    return _maker


class IndexField(LocatedField):
    def __init__(self, axis: common.Dimension, dtype: npt.DTypeLike) -> None:
        self.axis = axis
        self.dtype = np.dtype(dtype)

    def field_getitem(self, index: FieldIndexOrIndices) -> Any:
        if isinstance(index, int):
            return self.dtype.type(index)
        else:
            assert isinstance(index, tuple) and len(index) == 1 and isinstance(index[0], int)
            return self.dtype.type(index[0])

    @property
    def axes(self) -> tuple[common.Dimension]:
        return (self.axis,)


def index_field(axis: common.Dimension, dtype: npt.DTypeLike = int) -> LocatedField:
    return IndexField(axis, dtype)


class ConstantField(LocatedField):
    def __init__(self, value: Any, dtype: npt.DTypeLike):
        self.value = value
        self.dtype = np.dtype(dtype).type

    def field_getitem(self, _: FieldIndexOrIndices) -> Any:
        return self.dtype(self.value)

    @property
    def axes(self) -> tuple[()]:
        return ()


def constant_field(value: Any, dtype: Optional[npt.DTypeLike] = None) -> LocatedField:
    if dtype is None:
        dtype = infer_dtype_like_type(value)
    return ConstantField(value, dtype)


@builtins.shift.register(EMBEDDED)
def shift(*offsets: Union[runtime.Offset, int]) -> Callable[[ItIterator], ItIterator]:
    def impl(it: ItIterator) -> ItIterator:
        return it.shift(*list(o.value if isinstance(o, runtime.Offset) else o for o in offsets))

    return impl


DT = TypeVar("DT")


class _List(tuple, Generic[DT]):
    ...


@dataclasses.dataclass(frozen=True)
class _ConstList(Generic[DT]):
    value: DT

    def __getitem__(self, _):
        return self.value


@builtins.neighbors.register(EMBEDDED)
def neighbors(offset: runtime.Offset, it: ItIterator) -> _List:
    offset_str = offset.value if isinstance(offset, runtime.Offset) else offset
    assert isinstance(offset_str, str)
    offset_provider = offset_provider_cvar.get()
    assert offset_provider is not None
    connectivity = offset_provider[offset_str]
    assert isinstance(connectivity, common.Connectivity)
    return _List(
        shifted.deref()
        for i in range(connectivity.max_neighbors)
        if (shifted := it.shift(offset_str, i)).can_deref()
    )


@builtins.list_get.register(EMBEDDED)
def list_get(i, lst: _List[Optional[DT]]) -> Optional[DT]:
    return lst[i]


@builtins.map_.register(EMBEDDED)
def map_(op):
    def impl_(*lists):
        return _List(map(lambda x: op(*x), zip(*lists)))

    return impl_


@builtins.make_const_list.register(EMBEDDED)
def make_const_list(value):
    return _ConstList(value)


@builtins.reduce.register(EMBEDDED)
def reduce(fun, init):
    def sten(*lists):
        # TODO: assert check_that_all_lists_are_compatible(*lists)
        lst = None
        for cur in lists:
            if isinstance(cur, _List):
                lst = cur
                break
        # we can check a single argument for length,
        # because all arguments share the same pattern
        n = len(lst)
        res = init
        for i in range(n):
            res = fun(
                res,
                *(lst[i] for lst in lists),
            )
        return res

    return sten


@dataclasses.dataclass(frozen=True)
class SparseListIterator:
    it: ItIterator
    list_offset: Tag
    offsets: Sequence[OffsetPart] = dataclasses.field(default_factory=list, kw_only=True)

    def deref(self) -> Any:
        offset_provider = offset_provider_cvar.get()
        assert offset_provider is not None
        connectivity = offset_provider[self.list_offset]
        assert isinstance(connectivity, common.Connectivity)
        return _List(
            shifted.deref()
            for i in range(connectivity.max_neighbors)
            if (shifted := self.it.shift(*self.offsets, SparseTag(self.list_offset), i)).can_deref()
        )

    def can_deref(self) -> bool:
        return self.it.shift(*self.offsets).can_deref()

    def shift(self, *offsets: OffsetPart) -> SparseListIterator:
        return SparseListIterator(self.it, self.list_offset, offsets=[*offsets, *self.offsets])


@dataclasses.dataclass(frozen=True)
class ColumnDescriptor:
    axis: str
    col_range: range  # TODO(havogt) introduce range type that doesn't have step


@dataclasses.dataclass(frozen=True)
class ScanArgIterator:
    wrapped_iter: ItIterator
    k_pos: int

    def deref(self) -> Any:
        if not self.can_deref():
            return _UNDEFINED
        return self.wrapped_iter.deref()[self.k_pos]

    def can_deref(self) -> bool:
        return self.wrapped_iter.can_deref()

    def shift(self, *offsets: OffsetPart) -> ScanArgIterator:
        return ScanArgIterator(self.wrapped_iter.shift(*offsets), self.k_pos)


def shifted_scan_arg(k_pos: int) -> Callable[[ItIterator], ScanArgIterator]:
    def impl(it: ItIterator) -> ScanArgIterator:
        return ScanArgIterator(it, k_pos=k_pos)

    return impl


def is_located_field(field: Any) -> bool:
    return isinstance(field, LocatedField)  # TODO(havogt): avoid isinstance on Protocol


def has_uniform_tuple_element(field) -> bool:
    return field.dtype.fields is not None and all(
        next(iter(field.dtype.fields))[0] == f[0] for f in iter(field.dtype.fields)
    )


def is_tuple_of_field(field) -> bool:
    return isinstance(field, tuple) and all(
        is_located_field(f) or is_tuple_of_field(f) for f in field
    )


def is_field_of_tuple(field) -> bool:
    return is_located_field(field) and has_uniform_tuple_element(field)


def can_be_tuple_field(field) -> bool:
    return is_tuple_of_field(field) or is_field_of_tuple(field)


class TupleFieldMeta(type):
    def __instancecheck__(self, arg):
        return super().__instancecheck__(arg) or is_field_of_tuple(arg)


class TupleField(metaclass=TupleFieldMeta):
    """Allows uniform access to field of tuples and tuple of fields."""

    pass


def _get_axeses(field):
    if isinstance(field, tuple):
        return tuple(itertools.chain(*tuple(_get_axeses(f) for f in field)))
    else:
        assert is_located_field(field)
        return (field.axes,)


def _build_tuple_result(field, indices):
    if isinstance(field, tuple):
        return tuple(_build_tuple_result(f, indices) for f in field)
    else:
        assert is_located_field(field)
        return field[indices]


def _tuple_assign(field, value, indices):
    if isinstance(field, tuple):
        if len(field) != len(value):
            raise RuntimeError(
                f"Tuple of incompatible size, expected tuple of len={len(field)}, got len={len(value)}"
            )
        for f, v in zip(field, value):
            _tuple_assign(f, v, indices)
    else:
        assert is_located_field(field)
        field[indices] = value


class TupleOfFields(TupleField):
    def __init__(self, data):
        if not is_tuple_of_field(data):
            raise TypeError("Can only be instantiated with a tuple of fields")
        self.data = data
        axeses = _get_axeses(data)
        self.axes = axeses[0]

    def field_getitem(self, indices):
        return _build_tuple_result(self.data, indices)

    def field_setitem(self, indices, value):
        if not isinstance(value, tuple):
            raise RuntimeError(f"Value needs to be tuple, got `{value}`.")

        _tuple_assign(self.data, value, indices)


def as_tuple_field(field):
    assert can_be_tuple_field(field)

    if is_tuple_of_field(field):
        return TupleOfFields(field)

    assert isinstance(field, TupleField)  # e.g. field of tuple is already TupleField
    return field


def _column_dtype(elem: Any) -> np.dtype:
    if isinstance(elem, tuple):
        return np.dtype([(f"f{i}", _column_dtype(e)) for i, e in enumerate(elem)])
    else:
        return np.dtype(type(elem))


@builtins.scan.register(EMBEDDED)
def scan(scan_pass, is_forward: bool, init):
    def impl(*iters: ItIterator):
        column_range = column_range_cvar.get()
        if column_range is None:
            raise RuntimeError("Column range is not defined, cannot scan.")

        sorted_column_range = column_range if is_forward else reversed(column_range)
        state = init
        col = Column(column_range.start, np.zeros(len(column_range), dtype=_column_dtype(init)))
        for i in sorted_column_range:
            state = scan_pass(state, *map(shifted_scan_arg(i), iters))
            col[i] = state

        return col

    return impl


def _dimension_to_tag(domain: Domain) -> dict[Tag, range]:
    return {k.value if isinstance(k, common.Dimension) else k: v for k, v in domain.items()}


def _validate_domain(domain: Domain, offset_provider: OffsetProvider) -> None:
    if isinstance(domain, runtime.CartesianDomain):
        if any(isinstance(o, common.Connectivity) for o in offset_provider.values()):
            raise RuntimeError(
                "Got a `CartesianDomain`, but found a `Connectivity` in `offset_provider`, expected `UnstructuredDomain`."
            )


def fendef_embedded(fun: Callable[..., None], *args: Any, **kwargs: Any):
    if "offset_provider" not in kwargs:
        raise RuntimeError("offset_provider not provided")

    offset_provider = kwargs["offset_provider"]

    @runtime.closure.register(EMBEDDED)
    def closure(
        domain_: Domain,
        sten: Callable[..., Any],
        out: MutableLocatedField,
        ins: list[LocatedField],
    ) -> None:
        _validate_domain(domain_, kwargs["offset_provider"])
        domain: dict[Tag, range] = _dimension_to_tag(domain_)
        if not (is_located_field(out) or can_be_tuple_field(out)):
            raise TypeError("Out needs to be a located field.")

        column_range = None
        column: Optional[ColumnDescriptor] = None
        if kwargs.get("column_axis") and kwargs["column_axis"].value in domain:
            column_axis = kwargs["column_axis"]
            column = ColumnDescriptor(column_axis.value, domain[column_axis.value])
            del domain[column_axis.value]

            column_range = column.col_range

        out = as_tuple_field(out) if can_be_tuple_field(out) else out

        def _closure_runner():
            # Set context variables before executing the closure
            column_range_cvar.set(column_range)
            offset_provider_cvar.set(offset_provider)

            for pos in _domain_iterator(domain):
                promoted_ins = [promote_scalars(inp) for inp in ins]
                ins_iters = list(
                    make_in_iterator(
                        inp,
                        pos,
                        column_axis=column.axis if column else None,
                    )
                    for inp in promoted_ins
                )
                res = sten(*ins_iters)

                if column is None:
                    assert _is_concrete_position(pos)
                    ordered_indices = get_ordered_indices(out.axes, pos)
                    out.field_setitem(ordered_indices, res)
                else:
                    col_pos = pos.copy()
                    for k in column.col_range:
                        col_pos[column.axis] = k
                        assert _is_concrete_position(col_pos)
                        ordered_indices = get_ordered_indices(out.axes, col_pos)
                        out.field_setitem(ordered_indices, res[k])

        ctx = cvars.copy_context()
        ctx.run(_closure_runner)

    fun(*args)


runtime.fendef_embedded = fendef_embedded
