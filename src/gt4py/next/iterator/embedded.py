# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
import copy
import dataclasses
import itertools
import math
from functools import cached_property
from typing import (
    Any,
    Callable,
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
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from gt4py.eve import extended_typing as xtyping
from gt4py.next import common
from gt4py.next.iterator import builtins, runtime, utils


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
Scalar: TypeAlias = SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64


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


class MutableLocatedField(LocatedField, Protocol):
    """A LocatedField with write access."""

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_setitem(self, indices: FieldIndexOrIndices, value: Any) -> None:
        ...


_column_range: Optional[
    range
] = None  # TODO this is a bit ugly, alternative: pass scan range via iterator


class Column(np.lib.mixins.NDArrayOperatorsMixin):
    """Represents a column when executed in column mode (`column_axis != None`).

    Implements `__array_ufunc__` and `__array_function__` to isolate
    and simplify dispatching in iterator ir builtins.
    """

    def __init__(self, kstart: int, data: np.ndarray | Scalar) -> None:
        self.kstart = kstart
        assert isinstance(data, (np.ndarray, Scalar))  # type: ignore # mypy bug
        self.data = data if isinstance(data, np.ndarray) else np.full(len(_column_range), data)  # type: ignore[arg-type]

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

            @cached_property
            def incomplete_offsets(self):
                # TODO cleanup, test edge cases
                inherited_open_offsets = []
                for arg in self.args:
                    if arg.incomplete_offsets:
                        assert (
                            not inherited_open_offsets
                            or inherited_open_offsets == arg.incomplete_offsets
                        )
                        inherited_open_offsets = arg.incomplete_offsets
                # TODO: check order
                _, incomplete_offets = group_offsets(*inherited_open_offsets, *self.offsets)
                return incomplete_offets

            @cached_property
            def offset_provider(self):
                offset_provider = None
                for arg in self.args:
                    if new_offset_provider := arg.offset_provider:
                        offset_provider = new_offset_provider
                return offset_provider

            def _shifted_args(self):
                if not self.offsets:
                    return self.args
                return tuple(arg.shift(*self.offsets) for arg in self.args)

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


@builtins.reduce.register(EMBEDDED)
def reduce(fun, init, axis):
    def sten(*iters):
        # TODO: assert check_that_all_iterators_are_compatible(*iters)
        first_it = iters[0]
        res = init
        i = 0
        # we can check a single argument
        # because all arguments share the same pattern
        while builtins.can_deref(builtins.shift(axis, i)(first_it)):
            res = fun(
                res,
                *(builtins.deref(builtins.shift(axis, i)(it)) for it in iters),
            )
            i += 1
        return res

    return sten


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
    if np.issubdtype(val_type, np.number):
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
        if offset_implementation.value not in new_pos:
            new_pos[offset_implementation.value] = 0

        if is_int_index(value := new_pos[offset_implementation.value]):
            new_pos[offset_implementation.value] = value + index
        else:
            raise AssertionError()

        # If this is a local dimension limit the shift to `max_neighbors` of the respective
        # non-local dimensions connectivity, i.e. for V2EDim use `max_neighbors` of the V2E
        # connectivity. Otherwise a call to `reduce` of an iterator pointing to a sparse field
        # never returns as it can be shifted indefinitely.
        if offset_implementation.kind == common.DimensionKind.LOCAL:
            linked_offset_provider = offset_provider[offset_implementation.value]
            if not isinstance(linked_offset_provider, common.Connectivity):
                raise ValueError(
                    f"Offset provider for tag {offset_implementation.value} "
                    f"must be a `Connectivity`, but got {type(linked_offset_provider)}."
                )
            if new_pos[offset_implementation.value] >= linked_offset_provider.max_neighbors:
                return None
        return new_pos
    else:
        assert isinstance(offset_implementation, common.Connectivity)
        assert offset_implementation.origin_axis.value in pos
        new_pos = pos.copy()
        new_pos.pop(offset_implementation.origin_axis.value)
        cur_index = pos[offset_implementation.origin_axis.value]
        assert is_int_index(cur_index)
        if index >= offset_implementation.max_neighbors:
            return None
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


# The following holds for shifts:
# shift(tag, index)(inp) -> full shift
# shift(tag)(inp) -> incomplete shift
# shift(index)(shift(tag)(inp)) -> full shift
# Therefore the following transformation holds
# shift(e2v,0)(shift(c2e,2)(cell_field)) #noqa E800
# = shift(0)(shift(e2v)(shift(2)(shift(c2e)(cell_field))))
# = shift(c2e, 2, e2v, 0)(cell_field)
# = shift(c2e,e2v,2,0)(cell_field) <-- v2c,e2c twice incomplete shift
# = shift(2,0)(shift(c2e,e2v)(cell_field))
# for implementations it means everytime we have an index, we can "execute" a concrete shift
def group_offsets(*offsets: OffsetPart) -> tuple[list[CompleteOffset], list[Tag]]:
    tag_stack = []
    complete_offsets = []
    for offset in offsets:
        if not isinstance(offset, (int, np.integer)):
            tag_stack.append(offset)
        else:
            assert tag_stack
            tag = tag_stack.pop()
            complete_offsets.append((tag, offset))
    return complete_offsets, tag_stack


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


def get_open_offsets(*offsets: OffsetPart) -> list[Tag]:
    return group_offsets(*offsets)[1]


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
) -> Sequence[common.Dimension]:  # arbitrary nesting of tuples of LocatedField
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
    if isinstance(field_or_tuple, tuple):
        if column_axis is not None:
            assert _column_range
            # construct a Column of tuples
            column_axis_idx = _axis_idx(_get_axes(field_or_tuple), column_axis)
            if column_axis_idx is None:
                column_axis_idx = -1  # field doesn't have the column index, e.g. ContantField
            first = tuple(
                _make_tuple(f, _single_vertical_idx(indices, column_axis_idx, _column_range.start))
                for f in field_or_tuple
            )
            col = Column(
                _column_range.start, np.zeros(len(_column_range), dtype=_column_dtype(first))
            )
            col[0] = first
            for i in _column_range[1:]:
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
            assert _column_range is not None
            return Column(_column_range.start, data)
        else:
            return data


def _axis_idx(axes: Sequence[common.Dimension], axis: Tag) -> Optional[int]:
    for i, a in enumerate(axes):
        if a.value == axis:
            return i
    return None


@dataclasses.dataclass(frozen=True)
class MDIterator:
    field: LocatedField
    pos: MaybePosition
    incomplete_offsets: Sequence[Tag] = dataclasses.field(default_factory=list, kw_only=True)
    offset_provider: OffsetProvider = dataclasses.field(kw_only=True)
    column_axis: Optional[Tag] = dataclasses.field(default=None, kw_only=True)

    def shift(self, *offsets: OffsetPart) -> MDIterator:
        complete_offsets, open_offsets = group_offsets(*self.incomplete_offsets, *offsets)
        return MDIterator(
            self.field,
            shift_position(self.pos, *complete_offsets, offset_provider=self.offset_provider),
            incomplete_offsets=open_offsets,
            offset_provider=self.offset_provider,
            column_axis=self.column_axis,
        )

    def can_deref(self) -> bool:
        return self.pos is not None

    def deref(self) -> Any:
        assert not self.incomplete_offsets
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
        if self.column_axis is not None:
            assert _column_range is not None
            k_pos = shifted_pos.pop(self.column_axis)
            assert isinstance(k_pos, int)
            # the following range describes a range in the field
            # (negative values are relative to the origin, not relative to the size)
            slice_column[self.column_axis] = range(k_pos, k_pos + len(_column_range))

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


def make_in_iterator(
    inp: LocatedField,
    pos: Position,
    offset_provider: OffsetProvider,
    *,
    column_axis: Optional[Tag],
) -> MDIterator:
    new_pos: Position = pos.copy()
    if column_axis is not None:
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        assert _column_range is not None
        new_pos[column_axis] = _column_range.start
    return MDIterator(
        inp,
        new_pos,
        incomplete_offsets=[],
        offset_provider=offset_provider,
        column_axis=column_axis,
    )


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
    ):
        self.getter = getter
        self._axes = axes
        self.setter = setter
        self.array = array
        self.dtype = dtype

    def __getitem__(self, indices: ArrayIndexOrIndices) -> Any:
        return self.array()[indices]

    # TODO in a stable implementation of the Field concept we should make this behavior the default behavior for __getitem__
    def field_getitem(self, indices: FieldIndexOrIndices) -> Any:
        indices = utils.tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices: ArrayIndexOrIndices, value: Any):
        self.array()[indices] = value

    def field_setitem(self, indices: FieldIndexOrIndices, value: Any):
        self.setter(indices, value)

    def __array__(self) -> np.ndarray:
        return self.array()

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
            indices = utils.tupelize(indices)
            a[_shift_field_indices(indices, offsets) if offsets else indices] = value

        def getter(indices):
            return a[_shift_field_indices(indices, offsets) if offsets else indices]

        return LocatedFieldImpl(
            getter,
            axes,
            dtype=a.dtype,
            setter=setter,
            array=a.__array__,
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
        self.dtype = np.dtype(dtype)

    def field_getitem(self, _: FieldIndexOrIndices) -> Any:
        return self.dtype.type(self.value)

    def __array__(self) -> np.ndarray:
        return np.array(self.value)

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


class IgnoreShiftIt:
    def __init__(self, tag: runtime.Offset, it):
        self.tag = tag
        self.it = it

    def shift(self, tag: runtime.Offset, index: IntIndex, *tail):
        assert isinstance(self.tag.value, str)
        if tag == self.tag.value:
            return self

        result = IgnoreShiftIt(self.tag, shift(tag, index)(self.it))
        if tail:
            return shift(*tail)(result)
        return result

    def __getattr__(self, item):
        return getattr(self.it, item)


@builtins.ignore_shift.register(EMBEDDED)
def ignore_shift(tag: runtime.Offset) -> Callable[[ItIterator], IgnoreShiftIt]:
    return lambda it: IgnoreShiftIt(tag, it)


class TranslateShiftIt:
    def __init__(self, tag: runtime.Offset, new_tag: runtime.Offset, it):
        self.tag = tag
        self.new_tag = new_tag
        self.it = it

    def shift(self, tag: Tag, index: IntIndex, *tail):
        assert isinstance(self.tag.value, str) and isinstance(self.new_tag.value, str)
        if tag == self.tag.value:
            tag = self.new_tag.value

        result = TranslateShiftIt(self.tag, self.new_tag, shift(tag, index)(self.it))
        if tail:
            return shift(*tail)(result)
        return result

    def __getattr__(self, item):
        return getattr(self.it, item)


@builtins.translate_shift.register(EMBEDDED)
def translate_shift(
    tag: runtime.Offset, new_tag: runtime.Offset
) -> Callable[[ItIterator], TranslateShiftIt]:
    return lambda it: TranslateShiftIt(tag, new_tag, it)


@dataclasses.dataclass(frozen=True)
class ColumnDescriptor:
    axis: str
    col_range: range  # TODO(havogt) introduce range type that doesn't have step


@dataclasses.dataclass(frozen=True)
class ScanArgIterator:
    wrapped_iter: ItIterator
    k_pos: int
    offsets: Sequence[OffsetPart] = dataclasses.field(default_factory=list, kw_only=True)

    def deref(self) -> Any:
        if not self.can_deref():
            return _UNDEFINED
        return self.wrapped_iter.deref()[self.k_pos]

    def can_deref(self) -> bool:
        return self.wrapped_iter.can_deref()

    def shift(self, *offsets: OffsetPart) -> ScanArgIterator:
        return ScanArgIterator(self.wrapped_iter, self.k_pos, offsets=[*offsets, *self.offsets])


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
        if _column_range is None:
            raise RuntimeError("Column range is not defined, cannot scan.")

        column_range = _column_range if is_forward else reversed(_column_range)
        state = init
        col = Column(_column_range.start, np.zeros(len(_column_range), dtype=_column_dtype(init)))
        for i in column_range:
            state = scan_pass(state, *map(shifted_scan_arg(i), iters))
            col[i] = state

        return col

    return impl


def _dimension_to_tag(domain: Domain) -> dict[Tag, range]:
    return {k.value if isinstance(k, common.Dimension) else k: v for k, v in domain.items()}


def _validate_domain(domain: Domain, offset_provider: OffsetProvider) -> None:
    pass
    # TODO(tehrengruber): instead of this check for shifts
    # if isinstance(domain, runtime.CartesianDomain):
    #    if any(isinstance(o, common.Connectivity) for o in offset_provider.values()):
    #        raise RuntimeError(
    #            "Got a `CartesianDomain`, but found a `Connectivity` in `offset_provider`, expected `UnstructuredDomain`."
    #        )


def fendef_embedded(fun: Callable[..., None], *args: Any, **kwargs: Any):
    if "offset_provider" not in kwargs:
        raise RuntimeError("offset_provider not provided")

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

        global _column_range
        column: Optional[ColumnDescriptor] = None
        if kwargs.get("column_axis") and kwargs["column_axis"].value in domain:
            column_axis = kwargs["column_axis"]
            column = ColumnDescriptor(column_axis.value, domain[column_axis.value])
            del domain[column_axis.value]

            _column_range = column.col_range

        out = as_tuple_field(out) if can_be_tuple_field(out) else out

        for pos in _domain_iterator(domain):
            promoted_ins = [promote_scalars(inp) for inp in ins]
            ins_iters = list(
                make_in_iterator(
                    inp,
                    pos,
                    kwargs["offset_provider"],
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

        _column_range = None

    fun(*args)


runtime.fendef_embedded = fendef_embedded
