from __future__ import annotations

import itertools
import numbers
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeGuard,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from functional import iterator
from functional.common import Dimension
from functional.iterator import builtins
from functional.iterator.runtime import Offset
from functional.iterator.utils import tupelize


EMBEDDED = "embedded"

# TODO mark private/public functions/classes
# e.g. by moving all public from this file to somewhere else


class SparseOffset(str):
    ...


class NeighborTableOffsetProvider:
    def __init__(
        self,
        tbl: npt.NDArray,  # TODO(havogt): define neighbor table concept
        origin_axis: Dimension,
        neighbor_axis: Dimension,
        max_neighbors: int,
        has_skip_values=True,
    ) -> None:
        self.tbl = tbl
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        assert not hasattr(tbl, "shape") or tbl.shape[1] == max_neighbors
        self.max_neighbors = max_neighbors
        self.has_skip_values = has_skip_values


# Atoms
Tag: TypeAlias = str
IntIndex: TypeAlias = int

FieldIndex: TypeAlias = int | slice  # TODO probably numpy index type?
FieldIndexOrIndices: TypeAlias = FieldIndex | tuple[FieldIndex, ...]

Axis = Dimension | None

# Offsets
AnyOffset: TypeAlias = Union[Tag, IntIndex]
CompleteOffset: TypeAlias = tuple[Tag, IntIndex]
Connectivity: TypeAlias = NeighborTableOffsetProvider
OffsetProviderElem: TypeAlias = Dimension | Connectivity
OffsetProvider: TypeAlias = dict[Tag, OffsetProviderElem]

# Positions
SparsePositionEntry = list[int]
IncompleteSparsePositionEntry = list[Optional[int]]
PositionEntry = IntIndex | SparsePositionEntry
IncompletePositionEntry = IntIndex | IncompleteSparsePositionEntry
ConcretePosition: TypeAlias = dict[Tag, PositionEntry]
IncompletePosition: TypeAlias = dict[Tag, IncompletePositionEntry]

Position: TypeAlias = Union[ConcretePosition, IncompletePosition]
#: A ``None`` position flags invalid not-a-neighbor results in neighbor-table lookups
MaybePosition: TypeAlias = Optional[Position]


@runtime_checkable
class ItIterator(Protocol):
    """
    Prototype for the Iterator concept of Iterator IR.

    `ItIterator` to avoid name clashes with `Iterator` from `typing` and `collections.abc`.
    """

    def shift(self, *offsets: AnyOffset) -> ItIterator:
        ...

    def max_neighbors(self) -> int:
        ...

    def can_deref(self) -> bool:
        ...

    def deref(self) -> Any:
        ...


@runtime_checkable
class LocatedField(Protocol):
    """A field with named dimensions providing read access."""

    @property
    def axes(self) -> tuple[Dimension, ...]:
        ...

    def __getitem__(self, indices: FieldIndexOrIndices) -> Any:
        ...


class AssignableLocatedField(LocatedField):
    """A LocatedField with write access."""

    def __setitem__(self, indices: FieldIndexOrIndices, value: Any) -> None:
        ...


@builtins.deref.register(EMBEDDED)
def deref(it):
    return it.deref()


@builtins.can_deref.register(EMBEDDED)
def can_deref(it):
    return it.can_deref()


@builtins.if_.register(EMBEDDED)
def if_(cond, t, f):
    return t if cond else f


@builtins.not_.register(EMBEDDED)
def not_(a):
    return not a


@builtins.and_.register(EMBEDDED)
def and_(a, b):
    return a and b


@builtins.or_.register(EMBEDDED)
def or_(a, b):
    return a or b


@builtins.tuple_get.register(EMBEDDED)
def tuple_get(i, tup):
    return tup[i]


@builtins.make_tuple.register(EMBEDDED)
def make_tuple(*args):
    return (*args,)


@builtins.lift.register(EMBEDDED)
def lift(stencil):
    def impl(*args):
        class wrap_iterator:
            def __init__(self, *, offsets: list[AnyOffset] = None, elem=None) -> None:
                self.offsets = offsets or []
                self.elem = elem

            # TODO needs to be supported by all iterators that represent tuples
            def __getitem__(self, index):
                return wrap_iterator(offsets=self.offsets, elem=index)

            def shift(self, *offsets: AnyOffset):
                return wrap_iterator(offsets=[*self.offsets, *offsets], elem=self.elem)

            def max_neighbors(self):
                # TODO cleanup, test edge cases
                open_offsets = get_open_offsets(*self.offsets)
                assert open_offsets
                return _get_connectivity(args[0].offset_provider, open_offsets[0]).max_neighbors

            def _shifted_args(self):
                return tuple(map(lambda arg: arg.shift(*self.offsets), args))

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
                    return stencil(*shifted_args)
                else:
                    return stencil(*shifted_args)[self.elem]

        return wrap_iterator()

    return impl


@builtins.reduce.register(EMBEDDED)
def reduce(fun, init):
    def sten(*iters):
        # TODO: assert check_that_all_iterators_are_compatible(*iters)
        first_it = iters[0]
        n = first_it.max_neighbors()
        res = init
        for i in range(n):
            # we can check a single argument
            # because all arguments share the same pattern
            if not builtins.can_deref(builtins.shift(i)(first_it)):
                break
            res = fun(
                res,
                *(builtins.deref(builtins.shift(i)(it)) for it in iters),
            )
        return res

    return sten


@builtins.domain.register(EMBEDDED)
def domain(*args):
    return dict(args)


@builtins.named_range.register(EMBEDDED)
def named_range(tag, start, end):
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


@builtins.eq.register(EMBEDDED)
def eq(first, second):
    return first == second


@builtins.greater.register(EMBEDDED)
def greater(first, second):
    return first > second


@builtins.less.register(EMBEDDED)
def less(first, second):
    return first < second


def _lookup_offset_provider(offset_provider: OffsetProvider, tag: Tag) -> OffsetProviderElem:
    if tag not in offset_provider:
        raise RuntimeError(f"Missing offset provider for `{tag}`")
    return offset_provider[tag]


def _get_connectivity(offset_provider: OffsetProvider, tag: Tag) -> Connectivity:
    if not isinstance(connectivity := _lookup_offset_provider(offset_provider, tag), Connectivity):
        raise RuntimeError(f"Expected a `Connectivity` for `{tag}`")
    return connectivity


def _named_range(axis: str, range_: Iterable[int]) -> Iterable[tuple[str, int]]:
    return ((axis, i) for i in range_)


def _domain_iterator(domain: dict[str, range]) -> Iterable[Position]:
    return (
        dict(elem)
        for elem in itertools.product(*(_named_range(axis, rang) for axis, rang in domain.items()))
    )


def execute_shift(
    pos: Position, tag: Tag, index: IntIndex, *, offset_provider: OffsetProvider
) -> MaybePosition:
    assert pos is not None
    if isinstance(tag, SparseOffset):
        new_pos = deepcopy(pos)

        cur_pos = new_pos[tag]
        assert isinstance(cur_pos, list)
        for i, p in reversed(
            list(enumerate(cur_pos))
        ):  # first shift applies to the last sparse dimensions of that axis type
            if p is None:
                cur_pos[i] = index
                break
        return new_pos

    assert tag in offset_provider
    offset_implementation = offset_provider[tag]
    if isinstance(offset_implementation, Dimension):
        assert offset_implementation.value in pos
        new_pos = pos.copy()
        _extracted_val_for_mypy_check = new_pos[offset_implementation.value]
        assert isinstance(_extracted_val_for_mypy_check, int)
        new_pos[offset_implementation.value] = _extracted_val_for_mypy_check + index
        return new_pos
    elif isinstance(offset_implementation, NeighborTableOffsetProvider):
        assert offset_implementation.origin_axis.value in pos
        new_pos = pos.copy()
        new_pos.pop(offset_implementation.origin_axis.value)
        if offset_implementation.tbl[pos[offset_implementation.origin_axis.value], index] is None:
            return None
        else:
            new_pos[offset_implementation.neighbor_axis.value] = int(
                offset_implementation.tbl[pos[offset_implementation.origin_axis.value], index]
            )
        return new_pos

    raise AssertionError()


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
def group_offsets(*offsets: AnyOffset) -> tuple[list[CompleteOffset], list[Tag]]:
    tag_stack = []
    complete_offsets = []
    for offset in offsets:
        if not isinstance(offset, int):
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


def get_open_offsets(*offsets: AnyOffset) -> list[Tag]:
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


def _get_axes(field_or_tuple: Union[LocatedField, tuple]) -> list[Dimension]:
    return list(
        _get_axes(field_or_tuple[0]) if isinstance(field_or_tuple, tuple) else field_or_tuple.axes
    )


def _make_tuple(field_or_tuple: Union[LocatedField, tuple], indices) -> tuple:
    if isinstance(field_or_tuple, tuple):
        return tuple(_make_tuple(f, indices) for f in field_or_tuple)
    else:
        return field_or_tuple[indices]


def _is_position_fully_defined(
    pos: Position,
) -> TypeGuard[ConcretePosition]:
    return all(
        isinstance(v, int) or (isinstance(v, list) and all(isinstance(e, int) for e in v))
        for v in pos.values()
    )


class MDIterator:
    def __init__(
        self,
        field: LocatedField,
        pos: MaybePosition,
        *,
        incomplete_offsets: Sequence[Tag] = None,
        offset_provider: OffsetProvider,
        column_axis: Tag = None,
    ) -> None:
        self.field = field
        self.pos = pos
        self.incomplete_offsets = incomplete_offsets or []
        self.offset_provider = offset_provider
        self.column_axis = column_axis

    def shift(self, *offsets: AnyOffset) -> MDIterator:
        complete_offsets, open_offsets = group_offsets(*self.incomplete_offsets, *offsets)
        return MDIterator(
            self.field,
            shift_position(self.pos, *complete_offsets, offset_provider=self.offset_provider),
            incomplete_offsets=open_offsets,
            offset_provider=self.offset_provider,
            column_axis=self.column_axis,
        )

    def max_neighbors(self) -> int:
        assert self.incomplete_offsets
        return _get_connectivity(self.offset_provider, self.incomplete_offsets[0]).max_neighbors

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

        if not all(axis.value in shifted_pos.keys() for axis in axes if axis is not None):
            raise IndexError("Iterator position doesn't point to valid location for its field.")
        slice_column = dict[Tag, FieldIndex]()
        if self.column_axis is not None:
            slice_column[self.column_axis] = slice(shifted_pos[self.column_axis], None)
            del shifted_pos[self.column_axis]

        assert _is_position_fully_defined(shifted_pos)
        ordered_indices = get_ordered_indices(
            axes,
            {**shifted_pos, **slice_column},
        )
        try:
            return _make_tuple(self.field, ordered_indices)
        except IndexError:
            return _UNDEFINED


def make_in_iterator(
    inp: LocatedField,
    pos: Position,
    offset_provider: OffsetProvider,
    *,
    column_axis: Optional[Tag],
) -> MDIterator:
    axes = _get_axes(inp)
    sparse_dimensions = list[Tag]()
    for axis in axes:
        if isinstance(axis, Offset):
            assert isinstance(axis.value, str)
            sparse_dimensions.append(axis.value)
        elif isinstance(axis, Dimension) and axis.local:
            # we just use the name of the axis to match the offset literal for now
            sparse_dimensions.append(axis.value)

    new_pos: Position = pos.copy()
    for sparse_dim in set(sparse_dimensions):
        init = [None] * sparse_dimensions.count(sparse_dim)
        new_pos[sparse_dim] = init  # type: ignore[assignment] # looks like mypy is confused
    if column_axis is not None:
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        new_pos[column_axis] = 0
    return MDIterator(
        inp,
        new_pos,
        incomplete_offsets=list(map(lambda x: SparseOffset(x), sparse_dimensions)),
        offset_provider=offset_provider,
        column_axis=column_axis,
    )


builtins.builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


class LocatedFieldImpl:
    """A Field with named dimensions/axes.

    Axis keys can be any objects that are hashable.
    """

    @property
    def axes(self) -> tuple[Dimension, ...]:
        return self._axes

    def __init__(
        self,
        getter: Callable[[FieldIndexOrIndices], Any],
        axes: tuple[Dimension, ...],
        dtype,
        *,
        setter: Optional[Callable[[FieldIndexOrIndices, Any], None]] = None,
        array: Optional[Callable[[], npt.NDArray]] = None,
    ):
        self.getter = getter
        self._axes = axes
        self.setter = setter
        self.array = array
        self.dtype = dtype

    def __getitem__(self, indices: FieldIndexOrIndices) -> Any:
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices: FieldIndexOrIndices, value: Any):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)

    def __array__(self) -> np.ndarray:
        if self.array is None:
            raise TypeError("__array__ not supported for this field")
        return self.array()

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array().shape


def _is_tuple_axis(axis: Axis):
    return axis is None


def get_ordered_indices(
    axes: Iterable[Axis], pos: Mapping[str, FieldIndex | SparsePositionEntry]
) -> tuple[FieldIndex, ...]:
    res = list[FieldIndex]()
    pos = deepcopy(pos)
    for axis in axes:
        if _is_tuple_axis(axis):
            res.append(slice(None))
        else:
            assert axis is not None
            assert axis.value in pos
            elem = pos[axis.value]
            if isinstance(elem, list):
                res.append(elem.pop(0))
            else:
                assert isinstance(elem, (int, slice))
                res.append(elem)
    return tuple(res)


def _tupsum(a, b):
    def combine_slice(s, t):
        is_slice = False
        if isinstance(s, slice):
            is_slice = True
            first = 0 if s.start is None else s.start
            assert s.step is None
            assert s.stop is None
        else:
            assert isinstance(s, numbers.Integral)
            first = s
        if isinstance(t, slice):
            is_slice = True
            second = 0 if t.start is None else t.start
            assert t.step is None
            assert t.stop is None
        else:
            assert isinstance(t, numbers.Integral)
            second = t
        start = first + second
        return slice(start, None) if is_slice else start

    return tuple(combine_slice(*i) for i in zip(a, b))


def np_as_located_field(
    *axes: Dimension, origin: Optional[dict[Dimension, int]] = None
) -> Callable[[np.ndarray], LocatedFieldImpl]:
    def _maker(a: np.ndarray) -> LocatedFieldImpl:
        if a.ndim != len(axes):
            raise TypeError("ndarray.ndim incompatible with number of given axes")

        if origin is not None:
            offsets = get_ordered_indices(axes, {k.value: v for k, v in origin.items()})
        else:
            offsets = tuple(0 for _ in axes)

        def setter(indices, value):
            indices = tupelize(indices)
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedFieldImpl(getter, axes, dtype=a.dtype, setter=setter, array=a.__array__)

    return _maker


def index_field(axis: Dimension, dtype: type = float) -> LocatedField:
    class IndexField:
        def __init__(self, axis: Dimension, dtype: type) -> None:
            self.axis = axis
            self.dtype = dtype

        def __getitem__(self, index: FieldIndexOrIndices):  # TODO return type
            assert isinstance(index, int) or (isinstance(index, tuple) and len(index) == 1)
            return dtype(index if isinstance(index, int) else index[0])

        @property
        def axes(self) -> tuple[Dimension]:
            return (self.axis,)

    return IndexField(axis, dtype)


def constant_field(value: Any, dtype: type = float) -> LocatedField:
    class ConstantField:
        def __init__(self, value, dtype):
            self.value = value
            self.dtype = dtype

        def __getitem__(self, _: FieldIndexOrIndices):  # TODO return type
            return value

        @property
        def axes(self) -> tuple[()]:
            return ()

    return ConstantField(value, dtype)


@builtins.shift.register(EMBEDDED)
def shift(*offsets: Union[Offset, int]) -> Callable[[ItIterator], ItIterator]:
    def impl(it: ItIterator) -> ItIterator:
        return it.shift(*list(o.value if isinstance(o, Offset) else o for o in offsets))

    return impl


@dataclass
class Column:
    axis: str
    col_range: range  # TODO(havogt) introduce range type that doesn't have step


class ScanArgIterator:
    def __init__(
        self, wrapped_iter: ItIterator, k_pos: int, *, offsets: Sequence[AnyOffset] = None
    ) -> None:
        self.wrapped_iter = wrapped_iter
        self.offsets = offsets or []
        self.k_pos = k_pos

    def deref(self) -> Any:
        if not self.can_deref():
            return _UNDEFINED
        return self.wrapped_iter.deref()[self.k_pos]

    def can_deref(self) -> bool:
        return self.wrapped_iter.can_deref()

    def shift(self, *offsets: AnyOffset) -> ScanArgIterator:
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
        if not all(axes == axeses[0] for axes in axeses):
            raise TypeError("All fields in the tuple need the same axes.")
        self.axes = axeses[0]

    def __getitem__(self, indices):
        return _build_tuple_result(self.data, indices)

    def __setitem__(self, indices, value):
        if not isinstance(value, tuple):
            raise RuntimeError("Value needs to be tuple.")

        _tuple_assign(self.data, value, indices)


def as_tuple_field(field):
    assert can_be_tuple_field(field)

    if is_tuple_of_field(field):
        return TupleOfFields(field)

    assert isinstance(field, TupleField)  # e.g. field of tuple is already TupleField
    return field


_column_range = None  # TODO this is a bit ugly, alternative: pass scan range via iterator


@builtins.scan.register(EMBEDDED)
def scan(scan_pass, is_forward: bool, init):
    def impl(*iters: ItIterator):
        if _column_range is None:
            raise RuntimeError("Column range is not defined, cannot scan.")

        column_range = _column_range
        if not is_forward:
            column_range = reversed(column_range)

        state = init
        col = []
        for i in column_range:
            state = scan_pass(
                state, *map(shifted_scan_arg(i), iters)
            )  # more generic scan returns state and result as 2 different things
            col.append(state)

        if not is_forward:
            col = np.flip(col)

        if isinstance(col[0], tuple):
            dtype = np.dtype([("", type(c)) for c in col[0]])
            return np.asarray(col, dtype=dtype)

        return np.asarray(col)

    return impl


def fendef_embedded(fun, *args, **kwargs):
    if "offset_provider" not in kwargs:
        raise RuntimeError("offset_provider not provided")

    @iterator.runtime.closure.register(EMBEDDED)
    def closure(
        domain_: dict[Union[str, Dimension], range],
        sten: Callable[..., Any],
        out: AssignableLocatedField,
        ins: list[LocatedField],
    ) -> None:
        domain: dict[str, range] = {
            k.value if isinstance(k, Dimension) else k: v for k, v in domain_.items()
        }
        if not (is_located_field(out) or can_be_tuple_field(out)):
            raise TypeError("Out needs to be a located field.")

        global _column_range
        column: Optional[Column] = None
        if "column_axis" in kwargs:
            column_axis = kwargs["column_axis"]
            column = Column(column_axis.value, domain[column_axis.value])
            del domain[column_axis.value]

            _column_range = column.col_range

        out = as_tuple_field(out) if can_be_tuple_field(out) else out

        for pos in _domain_iterator(domain):
            ins_iters = list(
                make_in_iterator(
                    inp,
                    pos,
                    kwargs["offset_provider"],
                    column_axis=column.axis if column is not None else None,
                )
                for inp in ins
            )
            res = sten(*ins_iters)

            if column is None:
                assert _is_position_fully_defined(pos)
                ordered_indices = get_ordered_indices(out.axes, pos)
                out[ordered_indices] = res
            else:
                col_pos = pos.copy()
                for k in column.col_range:
                    col_pos[column.axis] = k
                    assert _is_position_fully_defined(col_pos)
                    ordered_indices = get_ordered_indices(out.axes, col_pos)
                    out[ordered_indices] = res[k]

        _column_range = None

    fun(*args)


iterator.runtime.fendef_embedded = fendef_embedded
