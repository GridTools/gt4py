# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(havogt) move public definitions and make this module private

from __future__ import annotations

import abc
import copy
import dataclasses
import itertools
import math
import sys
import warnings

import numpy as np
import numpy.typing as npt

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Self,
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
from gt4py.next import common, field_utils
from gt4py.next.embedded import (
    context as embedded_context,
    exceptions as embedded_exceptions,
    operators,
)
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import builtins, runtime
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_specifications as ts, type_translation


try:
    import dace
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


EMBEDDED = "embedded"


# Atoms
Tag: TypeAlias = common.Tag

ArrayIndex: TypeAlias = slice | common.IntIndex
ArrayIndexOrIndices: TypeAlias = ArrayIndex | tuple[ArrayIndex, ...]

FieldIndex: TypeAlias = (
    range | slice | common.IntIndex
)  # A `range` FieldIndex can be negative indicating a relative position with respect to origin, not wrap-around semantics like `slice` TODO(havogt): remove slice here
FieldIndices: TypeAlias = tuple[FieldIndex, ...]
FieldIndexOrIndices: TypeAlias = FieldIndex | FieldIndices

FieldAxis: TypeAlias = common.Dimension
TupleAxis: TypeAlias = type[None]
Axis: TypeAlias = Union[FieldAxis, TupleAxis]
Scalar: TypeAlias = (
    SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64 | np.bool_
)


class SparseTag(Tag): ...


class NeighborTableOffsetProvider:
    def __init__(
        self,
        table: core_defs.NDArrayObject,
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

    def mapped_index(
        self, primary: common.IntIndex, neighbor_idx: common.IntIndex
    ) -> common.IntIndex:
        res = self.table[(primary, neighbor_idx)]
        assert common.is_int_index(res)
        return res

    if dace:
        # Extension of NeighborTableOffsetProvider adding SDFGConvertible support in GT4Py Programs
        def _dace_data_ptr(self) -> int:
            obj = self.table
            if dace.dtypes.is_array(obj):
                if hasattr(obj, "__array_interface__"):
                    return obj.__array_interface__["data"][0]
                if hasattr(obj, "__cuda_array_interface__"):
                    return obj.__cuda_array_interface__["data"][0]
            raise ValueError("Unsupported data container.")

        def _dace_descriptor(self) -> dace.data.Data:
            return dace.data.create_datadescriptor(self.table)
    else:

        def _dace_data_ptr(self) -> NoReturn:  # type: ignore[misc]
            raise NotImplementedError(
                "data_ptr is only supported when the 'dace' module is available."
            )

        def _dace_descriptor(self) -> NoReturn:  # type: ignore[misc]
            raise NotImplementedError(
                "__descriptor__ is only supported when the 'dace' module is available."
            )

    data_ptr = _dace_data_ptr
    __descriptor__ = _dace_descriptor


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

    def mapped_index(
        self, primary: common.IntIndex, neighbor_idx: common.IntIndex
    ) -> common.IntIndex:
        return primary * self.max_neighbors + neighbor_idx


# Offsets
OffsetPart: TypeAlias = Tag | common.IntIndex
CompleteOffset: TypeAlias = tuple[Tag, common.IntIndex]
OffsetProviderElem: TypeAlias = common.OffsetProviderElem
OffsetProvider: TypeAlias = common.OffsetProvider

# Positions
SparsePositionEntry = list[int]
IncompleteSparsePositionEntry: TypeAlias = list[Optional[int]]
PositionEntry: TypeAlias = SparsePositionEntry | common.IntIndex
IncompletePositionEntry: TypeAlias = IncompleteSparsePositionEntry | common.IntIndex
ConcretePosition: TypeAlias = dict[Tag, PositionEntry]
IncompletePosition: TypeAlias = dict[Tag, IncompletePositionEntry]

Position: TypeAlias = Union[ConcretePosition, IncompletePosition]
#: A ``None`` position flags invalid not-a-neighbor results in neighbor-table lookups
MaybePosition: TypeAlias = Optional[Position]

NamedFieldIndices: TypeAlias = Mapping[Tag, FieldIndex | SparsePositionEntry]


@runtime_checkable
class ItIterator(Protocol):
    """
    Prototype for the Iterator concept of Iterator IR.

    `ItIterator` to avoid name clashes with `Iterator` from `typing` and `collections.abc`.
    """

    def shift(self, *offsets: OffsetPart) -> ItIterator: ...

    def can_deref(self) -> bool: ...

    def deref(self) -> Any: ...


@runtime_checkable
class LocatedField(Protocol):
    """A field with named dimensions providing read access."""

    @property
    @abc.abstractmethod
    def dims(self) -> tuple[common.Dimension, ...]: ...

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_getitem(self, indices: NamedFieldIndices) -> Any: ...

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple([0] * len(self.dims))


@runtime_checkable
class MutableLocatedField(LocatedField, Protocol):
    """A LocatedField with write access."""

    # TODO(havogt): define generic Protocol to provide a concrete return type
    @abc.abstractmethod
    def field_setitem(self, indices: NamedFieldIndices, value: Any) -> None: ...


class Column(np.lib.mixins.NDArrayOperatorsMixin):
    """Represents a column when executed in column mode (`column_axis != None`).

    Implements `__array_ufunc__` and `__array_function__` to isolate
    and simplify dispatching in iterator ir builtins.
    """

    def __init__(self, kstart: int, data: np.ndarray | Scalar) -> None:
        self.kstart = kstart
        assert isinstance(data, (np.ndarray, Scalar))
        column_range: common.NamedRange = embedded_context.closure_column_range.get()
        self.data = (
            data if isinstance(data, np.ndarray) else np.full(len(column_range.unit_range), data)
        )

    @property
    def dtype(self) -> np.dtype:
        # not directly dtype of `self.data` as that might be a structured type containing `None`
        return _elem_dtype(self.data[self.kstart])

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
        if wrong_kstarts := (  # noqa: F841 [unused-variable]
            set(arg.kstart for arg in args if isinstance(arg, Column)) - {self.kstart}
        ):
            raise ValueError(
                "Incompatible 'Column.kstart': it should be '{self.kstart}' but found other values: {wrong_kstarts}."
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
        return np.logical_not(a)
    return not a


@builtins.gamma.register(EMBEDDED)
def gamma(a):
    gamma_ = np.vectorize(math.gamma)
    if isinstance(a, Column):
        return Column(kstart=a.kstart, data=gamma_(a.data))
    res = gamma_(a)
    assert res.ndim == 0
    return res.item()


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


CompositeOfScalarOrField: TypeAlias = Scalar | common.Field | tuple["CompositeOfScalarOrField", ...]


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
    elif isinstance(val, common.Field):
        return val
    val_type = infer_dtype_like_type(val)
    if isinstance(val, Scalar):
        return constant_field(val)
    else:
        raise ValueError(
            f"Expected a 'Field' or a number ('float', 'np.int64', ...), got '{val_type}'."
        )


for math_builtin_name in builtins.MATH_BUILTINS:
    python_builtins = {"int": int, "float": float, "bool": bool, "str": str}
    decorator = getattr(builtins, math_builtin_name).register(EMBEDDED)
    impl: Callable
    if math_builtin_name == "gamma":
        continue  # treated explicitly
    elif math_builtin_name in python_builtins:
        # TODO: Should potentially use numpy fixed size types to be consistent
        #   with compiled backends. Currently using Python types to preserve
        #   existing behaviour.
        impl = python_builtins[math_builtin_name]
    else:
        impl = getattr(np, math_builtin_name)

    globals()[math_builtin_name] = decorator(impl)


def _named_range(axis: str, range_: Iterable[int]) -> Iterable[CompleteOffset]:
    return ((axis, i) for i in range_)


def _domain_iterator(domain: dict[Tag, range]) -> Iterable[ConcretePosition]:
    return (
        dict(elem)
        for elem in itertools.product(*(_named_range(axis, rang) for axis, rang in domain.items()))
    )


def execute_shift(
    pos: Position, tag: Tag, index: common.IntIndex, *, offset_provider: OffsetProvider
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
                offset_implementation = offset_provider[tag]
                assert isinstance(offset_implementation, common.Connectivity)
                cur_index = pos[offset_implementation.origin_axis.value]
                assert common.is_int_index(cur_index)
                if offset_implementation.mapped_index(cur_index, index) in [
                    None,
                    common._DEFAULT_SKIP_VALUE,
                ]:
                    return None

                new_entry[i] = index
                break
        # the assertions above confirm pos is incomplete casting here to avoid duplicating work in a type guard
        return cast(IncompletePosition, pos) | {tag: new_entry}

    assert tag in offset_provider
    offset_implementation = offset_provider[tag]
    if isinstance(offset_implementation, common.Dimension):
        new_pos = copy.copy(pos)
        if common.is_int_index(value := new_pos[offset_implementation.value]):
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
        assert common.is_int_index(cur_index)
        if offset_implementation.mapped_index(cur_index, index) in [
            None,
            common._DEFAULT_SKIP_VALUE,
        ]:
            return None
        else:
            new_index = offset_implementation.mapped_index(cur_index, index)
            assert new_index is not None
            new_pos[offset_implementation.neighbor_axis.value] = int(new_index)

        return new_pos

    raise AssertionError("Unknown object in 'offset_provider'.")


def _is_list_of_complete_offsets(
    complete_offsets: list[tuple[Any, Any]],
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

    def __int__(self):
        return sys.maxsize

    def __repr__(self):
        return "_UNDEFINED"

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
    if isinstance(field_or_tuple, tuple):
        first = _get_axes(field_or_tuple[0])
        assert all(first == _get_axes(f) for f in field_or_tuple)
        return first
    else:
        return field_or_tuple.dims


def _single_vertical_idx(
    indices: NamedFieldIndices, column_axis: Tag, column_index: common.IntIndex
) -> NamedFieldIndices:
    transformed = {
        axis: (index if axis != column_axis else index.start + column_index)  # type: ignore[union-attr] # trust me, `index` is range in case of `column_axis` # fmt: off
        for axis, index in indices.items()
    }
    return transformed


@overload
def _make_tuple(
    field_or_tuple: tuple[tuple | LocatedField, ...],  # arbitrary nesting of tuples of Field
    named_indices: NamedFieldIndices,
    *,
    column_axis: Tag,
) -> tuple[tuple | Column, ...]: ...


@overload
def _make_tuple(
    field_or_tuple: tuple[tuple | LocatedField, ...],  # arbitrary nesting of tuples of Field
    named_indices: NamedFieldIndices,
    *,
    column_axis: Literal[None] = None,
) -> tuple[tuple | npt.DTypeLike | Undefined, ...]:  # arbitrary nesting
    ...


@overload
def _make_tuple(
    field_or_tuple: LocatedField, named_indices: NamedFieldIndices, *, column_axis: Tag
) -> Column: ...


@overload
def _make_tuple(
    field_or_tuple: LocatedField,
    named_indices: NamedFieldIndices,
    *,
    column_axis: Literal[None] = None,
) -> npt.DTypeLike | Undefined: ...


def _make_tuple(
    field_or_tuple: LocatedField | tuple[tuple | LocatedField, ...],
    named_indices: NamedFieldIndices,
    *,
    column_axis: Optional[Tag] = None,
) -> Column | npt.DTypeLike | tuple[tuple | Column | npt.DTypeLike | Undefined, ...] | Undefined:
    if column_axis is None:
        if isinstance(field_or_tuple, tuple):
            return tuple(_make_tuple(f, named_indices) for f in field_or_tuple)
        else:
            try:
                data = field_or_tuple.field_getitem(named_indices)
                return data
            except embedded_exceptions.IndexOutOfBounds:
                return _UNDEFINED
    else:
        column_range = embedded_context.closure_column_range.get().unit_range
        assert column_range is not None

        col: list[
            npt.DTypeLike | tuple[tuple | Column | npt.DTypeLike | Undefined, ...] | Undefined
        ] = []
        for i in column_range:
            # we don't know the buffer size, therefore we have to try.
            try:
                col.append(
                    tuple(
                        _make_tuple(
                            f,
                            _single_vertical_idx(
                                named_indices, column_axis, i - column_range.start
                            ),
                        )
                        for f in field_or_tuple
                    )
                    if isinstance(field_or_tuple, tuple)
                    else _make_tuple(
                        field_or_tuple,
                        _single_vertical_idx(named_indices, column_axis, i - column_range.start),
                    )
                )
            except embedded_exceptions.IndexOutOfBounds:
                col.append(_UNDEFINED)

        first = next((v for v in col if v != _UNDEFINED), None)
        if first is None:
            raise RuntimeError(
                "Found 'Undefined' value, this should not happen for a legal program."
            )
        dtype = _elem_dtype(first)
        return Column(column_range.start, np.asarray(col, dtype=dtype))


@dataclasses.dataclass(frozen=True)
class MDIterator:
    field: LocatedField | tuple[LocatedField | tuple, ...]  # arbitrary nesting
    pos: MaybePosition
    column_axis: Optional[Tag] = dataclasses.field(default=None, kw_only=True)

    def shift(self, *offsets: OffsetPart) -> MDIterator:
        complete_offsets = group_offsets(*offsets)
        offset_provider = embedded_context.offset_provider.get()
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
        if self.column_axis is not None:
            column_range = embedded_context.closure_column_range.get()
            assert column_range is not None
            k_pos = shifted_pos.pop(self.column_axis)
            assert isinstance(k_pos, int)
            # the following range describes a range in the field
            # (negative values are relative to the origin, not relative to the size)
            slice_column[self.column_axis] = range(k_pos, k_pos + len(column_range.unit_range))

        assert _is_concrete_position(shifted_pos)
        position = {**shifted_pos, **slice_column}
        return _make_tuple(self.field, position, column_axis=self.column_axis)


def _get_sparse_dimensions(axes: Sequence[common.Dimension]) -> list[Tag]:
    return [
        axis.value
        for axis in axes
        if isinstance(axis, common.Dimension) and axis.kind == common.DimensionKind.LOCAL
    ]


def _wrap_field(field: common.Field | tuple) -> NDArrayLocatedFieldWrapper | tuple:
    if isinstance(field, tuple):
        return tuple(_wrap_field(f) for f in field)
    else:
        assert isinstance(field, common.Field)
        return NDArrayLocatedFieldWrapper(field)


def make_in_iterator(
    inp_: common.Field, pos: Position, *, column_dimension: Optional[common.Dimension]
) -> ItIterator:
    inp = _wrap_field(inp_)
    axes = _get_axes(inp)
    sparse_dimensions = _get_sparse_dimensions(axes)
    new_pos: Position = pos.copy()
    for sparse_dim in set(sparse_dimensions):
        init = [None] * sparse_dimensions.count(sparse_dim)
        new_pos[sparse_dim] = init  # type: ignore[assignment] # looks like mypy is confused
    if column_dimension is not None:
        column_range = embedded_context.closure_column_range.get().unit_range
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        assert column_range is not None
        new_pos[column_dimension.value] = column_range.start
    it = MDIterator(
        inp, new_pos, column_axis=column_dimension.value if column_dimension is not None else None
    )
    if len(sparse_dimensions) >= 1:
        if len(sparse_dimensions) == 1:
            return SparseListIterator(it, sparse_dimensions[0])
        else:
            raise NotImplementedError(
                f"More than one local dimension is currently not supported, got {sparse_dimensions}."
            )
    else:
        return it


builtins.builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


@dataclasses.dataclass(frozen=True)
class NDArrayLocatedFieldWrapper(MutableLocatedField):
    """A temporary helper until we sorted out all Field conventions between frontend and iterator.embedded."""

    _ndarrayfield: common.Field

    @property
    def dims(self) -> tuple[common.Dimension, ...]:
        return self._ndarrayfield.__gt_domain__.dims

    def _translate_named_indices(
        self, _named_indices: NamedFieldIndices
    ) -> common.AbsoluteIndexSequence:
        named_indices: Mapping[common.Dimension, FieldIndex | SparsePositionEntry] = {
            d: _named_indices[d.value] for d in self._ndarrayfield.__gt_domain__.dims
        }
        domain_slice: list[common.NamedRange | common.NamedIndex] = []
        for d, v in named_indices.items():
            if isinstance(v, range):
                domain_slice.append(common.NamedRange(d, common.UnitRange(v.start, v.stop)))
            elif isinstance(v, list):
                assert len(v) == 1  # only 1 sparse dimension is supported
                assert common.is_int_index(
                    v[0]
                )  # derefing a concrete element in a sparse field, not a slice
                domain_slice.append(common.NamedIndex(d, v[0]))
            else:
                assert common.is_int_index(v)
                domain_slice.append(common.NamedIndex(d, v))
        return tuple(domain_slice)

    def field_getitem(self, named_indices: NamedFieldIndices) -> Any:
        return self._ndarrayfield[self._translate_named_indices(named_indices)].as_scalar()

    def field_setitem(self, named_indices: NamedFieldIndices, value: Any):
        if isinstance(self._ndarrayfield, common.MutableField):
            self._ndarrayfield[self._translate_named_indices(named_indices)] = value
        else:
            raise RuntimeError("Assigment into a non-mutable Field is not allowed.")

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return self._ndarrayfield.__gt_origin__


def _is_field_axis(axis: Axis) -> TypeGuard[FieldAxis]:
    return isinstance(axis, FieldAxis)


def _is_tuple_axis(axis: Axis) -> TypeGuard[TupleAxis]:
    return axis is None


def _is_sparse_position_entry(
    pos: FieldIndex | SparsePositionEntry,
) -> TypeGuard[SparsePositionEntry]:
    return isinstance(pos, list)


def get_ordered_indices(axes: Iterable[Axis], pos: NamedFieldIndices) -> tuple[FieldIndex, ...]:
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
def _shift_range(range_or_index: range, offset: int) -> slice: ...


@overload
def _shift_range(range_or_index: common.IntIndex, offset: int) -> common.IntIndex: ...


def _shift_range(range_or_index: range | common.IntIndex, offset: int) -> ArrayIndex:
    if isinstance(range_or_index, range):
        # range_or_index describes a range in the field
        assert range_or_index.step == 1
        return slice(range_or_index.start + offset, range_or_index.stop + offset)
    else:
        assert common.is_int_index(range_or_index)
        return range_or_index + offset


@overload
def _range2slice(r: range) -> slice: ...


@overload
def _range2slice(r: common.IntIndex) -> common.IntIndex: ...


def _range2slice(r: range | common.IntIndex) -> slice | common.IntIndex:
    if isinstance(r, range):
        assert r.start >= 0 and r.stop >= r.start
        return slice(r.start, r.stop)
    return r


def _shift_field_indices(
    ranges_or_indices: tuple[range | common.IntIndex, ...], offsets: tuple[int, ...]
) -> tuple[ArrayIndex, ...]:
    return tuple(
        _range2slice(r) if o == 0 else _shift_range(r, o)
        for r, o in zip(ranges_or_indices, offsets)
    )


def np_as_located_field(
    *axes: common.Dimension, origin: Optional[dict[common.Dimension, int]] = None
) -> Callable[[np.ndarray], common.Field]:
    warnings.warn("`np_as_located_field()` is deprecated, use `gtx.as_field()`", DeprecationWarning)  # noqa: B028 [no-explicit-stacklevel]

    origin = origin or {}

    def _maker(a) -> common.Field:
        if a.ndim != len(axes):
            raise TypeError("'ndarray.ndim' is incompatible with number of given dimensions.")
        ranges = []
        for d, s in zip(axes, a.shape):
            offset = origin.get(d, 0)
            ranges.append(common.UnitRange(-offset, s - offset))

        res = common._field(a, domain=common.Domain(dims=tuple(axes), ranges=tuple(ranges)))
        return res

    return _maker


@dataclasses.dataclass(frozen=True)
class IndexField(common.Field):
    """
    Minimal index field implementation.

    TODO: Improve implementation (e.g. support slicing) and move out of this module.
    """

    _dimension: common.Dimension
    _cur_index: Optional[core_defs.IntegralScalar] = None

    @property
    def __gt_domain__(self) -> common.Domain:
        return self.domain

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return (0,)

    @classmethod
    def __gt_builtin_func__(func: Callable, /) -> NoReturn:  # type: ignore[override] # Signature incompatible with supertype # fmt: off
        raise NotImplementedError()

    @property
    def domain(self) -> common.Domain:
        if self._cur_index is None:
            return common.Domain(common.NamedRange(self._dimension, common.UnitRange.infinite()))
        else:
            return common.Domain()

    @property
    def codomain(self) -> type[core_defs.int32]:
        return core_defs.int32

    @property
    def dtype(self) -> core_defs.Int32DType:
        return core_defs.Int32DType()

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        raise AttributeError("Cannot get 'ndarray' of an infinite 'Field'.")

    def asnumpy(self) -> np.ndarray:
        raise NotImplementedError()

    def as_scalar(self) -> core_defs.IntegralScalar:
        if self.domain.ndim != 0:
            raise ValueError(
                "'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        assert self._cur_index is not None
        return self._cur_index

    def premap(self, index_field: common.ConnectivityField | fbuiltins.FieldOffset) -> common.Field:
        # TODO can be implemented by constructing and ndarray (but do we know of which kind?)
        raise NotImplementedError()

    def restrict(self, item: common.AnyIndexSpec) -> Self:
        if isinstance(item, Sequence) and all(isinstance(e, common.NamedIndex) for e in item):
            assert isinstance(item[0], common.NamedIndex)  # for mypy errors on multiple lines below
            d, r = item[0]
            assert d == self._dimension
            assert isinstance(r, core_defs.INTEGRAL_TYPES)
            return self.__class__(self._dimension, r)
        # TODO set a domain...
        raise NotImplementedError()

    __call__ = premap
    __getitem__ = restrict

    def __abs__(self) -> common.Field:
        raise NotImplementedError()

    def __neg__(self) -> common.Field:
        raise NotImplementedError()

    def __invert__(self) -> common.Field:
        raise NotImplementedError()

    def __eq__(self, other: Any) -> common.Field:  # type: ignore[override] # mypy wants return `bool`
        raise NotImplementedError()

    def __ne__(self, other: Any) -> common.Field:  # type: ignore[override] # mypy wants return `bool`
        raise NotImplementedError()

    def __add__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __radd__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __sub__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rsub__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __mul__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rmul__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __floordiv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rfloordiv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __truediv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rtruediv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __pow__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __and__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __or__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __xor__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()


def index_field(axis: common.Dimension) -> common.Field:
    return IndexField(axis)


@dataclasses.dataclass(frozen=True)
class ConstantField(common.Field[Any, core_defs.ScalarT]):
    """
    Minimal constant field implementation.

    TODO: Improve implementation (e.g. support slicing) and move out of this module.
    """

    _value: core_defs.ScalarT

    @property
    def __gt_domain__(self) -> common.Domain:
        return self.domain

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple()

    @classmethod
    def __gt_builtin_func__(func: Callable, /) -> NoReturn:  # type: ignore[override] # Signature incompatible with supertype # fmt: off
        raise NotImplementedError()

    @property
    def domain(self) -> common.Domain:
        return common.Domain(dims=(), ranges=())

    @property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(type(self._value))

    @property
    def codomain(self) -> type[core_defs.ScalarT]:
        return self.dtype.scalar_type

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        raise AttributeError("Cannot get 'ndarray' of an infinite 'Field'.")

    def asnumpy(self) -> np.ndarray:
        raise NotImplementedError()

    def premap(self, index_field: common.ConnectivityField | fbuiltins.FieldOffset) -> common.Field:
        # TODO can be implemented by constructing and ndarray (but do we know of which kind?)
        raise NotImplementedError()

    def restrict(self, item: common.AnyIndexSpec) -> Self:
        # TODO set a domain...
        return self

    def as_scalar(self) -> core_defs.ScalarT:
        assert self.domain.ndim == 0
        return self._value

    __call__ = premap
    __getitem__ = restrict

    def __abs__(self) -> common.Field:
        raise NotImplementedError()

    def __neg__(self) -> common.Field:
        raise NotImplementedError()

    def __invert__(self) -> common.Field:
        raise NotImplementedError()

    def __eq__(self, other: Any) -> common.Field:  # type: ignore[override] # mypy wants return `bool`
        raise NotImplementedError()

    def __ne__(self, other: Any) -> common.Field:  # type: ignore[override] # mypy wants return `bool`
        raise NotImplementedError()

    def __add__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __radd__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __sub__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rsub__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __mul__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rmul__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __floordiv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rfloordiv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __truediv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __rtruediv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __pow__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __and__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __or__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()

    def __xor__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        raise NotImplementedError()


def constant_field(value: Any, dtype_like: Optional[core_defs.DTypeLike] = None) -> common.Field:
    if dtype_like is None:
        dtype_like = infer_dtype_like_type(value)
    dtype = core_defs.dtype(dtype_like)
    return ConstantField(dtype.scalar_type(value))


@builtins.shift.register(EMBEDDED)
def shift(*offsets: Union[runtime.Offset, int]) -> Callable[[ItIterator], ItIterator]:
    def impl(it: ItIterator) -> ItIterator:
        return it.shift(*list(o.value if isinstance(o, runtime.Offset) else o for o in offsets))

    return impl


DT = TypeVar("DT")


class _List(tuple, Generic[DT]): ...


@dataclasses.dataclass(frozen=True)
class _ConstList(Generic[DT]):
    value: DT

    def __getitem__(self, _):
        return self.value


@builtins.neighbors.register(EMBEDDED)
def neighbors(offset: runtime.Offset, it: ItIterator) -> _List:
    offset_str = offset.value if isinstance(offset, runtime.Offset) else offset
    assert isinstance(offset_str, str)
    offset_provider = embedded_context.offset_provider.get()
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
            res = fun(res, *(lst[i] for lst in lists))
        return res

    return sten


@dataclasses.dataclass(frozen=True)
class SparseListIterator:
    it: ItIterator
    list_offset: Tag
    offsets: Sequence[OffsetPart] = dataclasses.field(default_factory=list, kw_only=True)

    def deref(self) -> Any:
        offset_provider = embedded_context.offset_provider.get()
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
        return ScanArgIterator(it, k_pos=k_pos)  # here we evaluate the full column in every step

    return impl


def is_located_field(field: Any) -> TypeGuard[LocatedField]:
    return isinstance(field, LocatedField)


def is_mutable_located_field(field: Any) -> TypeGuard[MutableLocatedField]:
    return isinstance(field, MutableLocatedField)


def is_tuple_of_field(field) -> bool:
    return isinstance(field, tuple) and all(
        isinstance(f, common.Field) or is_tuple_of_field(f) for f in field
    )


class TupleFieldMeta(type): ...


class TupleField(metaclass=TupleFieldMeta):
    """Allows uniform access to field of tuples and tuple of fields."""

    pass


def _build_tuple_result(field: tuple | LocatedField, named_indices: NamedFieldIndices) -> Any:
    if isinstance(field, tuple):
        return tuple(_build_tuple_result(f, named_indices) for f in field)
    else:
        assert is_located_field(field)
        return field.field_getitem(named_indices)


def _tuple_assign(field: tuple | MutableLocatedField, value: Any, named_indices: NamedFieldIndices):
    if isinstance(field, tuple):
        if len(field) != len(value):
            raise RuntimeError(
                f"Tuple of incompatible size, expected tuple of 'len={len(field)}', got 'len={len(value)}'."
            )
        for f, v in zip(field, value):
            _tuple_assign(f, v, named_indices)
    else:
        assert is_mutable_located_field(field)
        field.field_setitem(named_indices, value)


class TupleOfFields(TupleField):
    def __init__(self, data):
        self.data = data
        self.dims = _get_axes(data)

    def field_getitem(self, named_indices: NamedFieldIndices) -> Any:
        return _build_tuple_result(self.data, named_indices)

    def field_setitem(self, named_indices: NamedFieldIndices, value: Any):
        if not isinstance(value, tuple):
            raise RuntimeError(f"Value needs to be tuple, got '{value}'.")
        _tuple_assign(self.data, value, named_indices)


def as_tuple_field(field: tuple | TupleField) -> TupleField:
    assert is_tuple_of_field(field)
    assert not isinstance(field, TupleField)
    return TupleOfFields(tuple(_wrap_field(f) for f in field))


def _elem_dtype(elem: Any) -> np.dtype:
    if hasattr(elem, "dtype"):
        return elem.dtype
    if isinstance(elem, tuple):
        return np.dtype([(f"f{i}", _elem_dtype(e)) for i, e in enumerate(elem)])
    return np.dtype(type(elem))


@builtins.scan.register(EMBEDDED)
def scan(scan_pass, is_forward: bool, init):
    def impl(*iters: ItIterator):
        column_range = embedded_context.closure_column_range.get().unit_range
        if column_range is None:
            raise RuntimeError("Column range is not defined, cannot scan.")

        sorted_column_range = column_range if is_forward else reversed(column_range)
        state = init
        col = Column(column_range.start, np.zeros(len(column_range), dtype=_elem_dtype(init)))
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
                "Got a 'CartesianDomain', but found a 'Connectivity' in 'offset_provider', expected 'UnstructuredDomain'."
            )


@runtime.set_at.register(EMBEDDED)
def set_at(expr: common.Field, domain: common.DomainLike, target: common.MutableField) -> None:
    operators._tuple_assign_field(target, expr, common.domain(domain))


def _compute_at_position(
    sten: Callable,
    ins: Sequence[common.Field],
    pos: ConcretePosition,
    column_dimension: Optional[common.Dimension],
) -> Scalar | tuple[Scalar | tuple, ...]:
    ins_iters = list(
        make_in_iterator(
            inp,
            pos,
            column_dimension=column_dimension,
        )
        for inp in ins
    )
    return sten(*ins_iters)


def _extract_column_range(domain) -> common.NamedRange | eve.NothingType:
    if (col_range_placeholder := embedded_context.closure_column_range.get(None)) is not None:
        assert (
            col_range_placeholder.unit_range.is_empty()
        )  # check it's just the placeholder with empty range
        column_axis = col_range_placeholder.dim
        if column_axis is not None and column_axis.value in domain:
            return common.NamedRange(
                column_axis,
                common.UnitRange(domain[column_axis.value].start, domain[column_axis.value].stop),
            )
    return eve.NOTHING


def _structured_dtype_to_typespec(structured_dtype: np.dtype) -> ts.ScalarType | ts.TupleType:
    if structured_dtype.names is None:
        return type_translation.from_dtype(core_defs.dtype(structured_dtype))
    return ts.TupleType(
        types=[
            _structured_dtype_to_typespec(structured_dtype[name]) for name in structured_dtype.names
        ]
    )


def _get_output_type(
    fun: Callable,
    domain_: runtime.CartesianDomain | runtime.UnstructuredDomain,
    args: tuple[Any, ...],
) -> ts.TypeSpec:
    domain = _dimension_to_tag(domain_)
    col_range = _extract_column_range(domain)

    col_dim: Optional[common.Dimension] = None
    if isinstance(col_range, common.NamedRange):
        col_dim = col_range.dim
        del domain[col_range.dim.value]

    # determine dtype by computing result at one point
    pos_in_domain = next(iter(_domain_iterator(domain)))
    with embedded_context.new_context(closure_column_range=col_range) as ctx:
        single_pos_result = ctx.run(_compute_at_position, fun, args, pos_in_domain, col_dim)
    assert single_pos_result is not _UNDEFINED, "Stencil contains an Out-Of-Bound access."
    dtype = _elem_dtype(single_pos_result)
    return _structured_dtype_to_typespec(dtype)


@builtins.as_fieldop.register(EMBEDDED)
def as_fieldop(fun: Callable, domain: runtime.CartesianDomain | runtime.UnstructuredDomain):
    def impl(*args):
        xp = field_utils.get_array_ns(*args)
        type_ = _get_output_type(fun, domain, [promote_scalars(arg) for arg in args])
        out = field_utils.field_from_typespec(type_, common.domain(domain), xp)

        # TODO(havogt): after updating all tests to use the new program,
        # we should get rid of closure and move the implementation to this function
        closure(_dimension_to_tag(domain), fun, out, list(args))
        return out

    return impl


@runtime.closure.register(EMBEDDED)
def closure(
    domain_: Domain,
    sten: Callable[..., Any],
    out,  #: MutableLocatedField,
    ins: list[common.Field | Scalar | tuple[common.Field | Scalar | tuple, ...]],
) -> None:
    assert embedded_context.within_valid_context()
    offset_provider = embedded_context.offset_provider.get()
    _validate_domain(domain_, offset_provider)
    domain: dict[Tag, range] = _dimension_to_tag(domain_)
    if not (isinstance(out, common.Field) or is_tuple_of_field(out)):
        raise TypeError("'Out' needs to be a located field.")

    column_range: common.NamedRange | eve.NothingType = _extract_column_range(domain)

    column_dim = None
    if isinstance(column_range, common.NamedRange):
        column_dim = column_range.dim
        del domain[column_range.dim.value]

    out = as_tuple_field(out) if is_tuple_of_field(out) else _wrap_field(out)
    promoted_ins = [promote_scalars(inp) for inp in ins]

    with embedded_context.new_context(closure_column_range=column_range) as ctx:

        def _iterate():
            for pos in _domain_iterator(domain):
                res = _compute_at_position(sten, promoted_ins, pos, column_dim)

                if column_range is eve.NOTHING:
                    assert _is_concrete_position(pos)
                    out.field_setitem(pos, res)
                else:
                    col_pos = pos.copy()
                    for k in column_range.unit_range:
                        col_pos[column_range.dim.value] = k
                        assert _is_concrete_position(col_pos)
                        out.field_setitem(col_pos, res[k])

        ctx.run(_iterate)


def fendef_embedded(fun: Callable[..., None], *args: Any, **kwargs: Any):
    if "offset_provider" not in kwargs:
        raise RuntimeError("'offset_provider' not provided.")

    context_vars = {"offset_provider": kwargs["offset_provider"]}
    if "column_axis" in kwargs:
        context_vars["closure_column_range"] = common.NamedRange(
            kwargs["column_axis"],
            common.UnitRange(0, 0),  # empty: indicates column operation, will update later
        )

    import inspect

    if len(args) < len(inspect.getfullargspec(fun).args):
        args = (*args, *arguments.iter_size_args(args))

    with embedded_context.new_context(**context_vars) as ctx:
        ctx.run(fun, *args)


runtime.fendef_embedded = fendef_embedded
