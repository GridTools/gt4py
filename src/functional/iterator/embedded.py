import itertools
import numbers
from dataclasses import dataclass

import numpy as np

from functional import iterator
from functional.iterator import builtins
from functional.iterator.runtime import CartesianAxis, Offset
from functional.iterator.utils import tupelize


EMBEDDED = "embedded"


class NeighborTableOffsetProvider:
    def __init__(self, tbl, origin_axis, neighbor_axis, max_neighbors) -> None:
        self.tbl = tbl
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        self.max_neighbors = max_neighbors


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
            def __init__(self, *, offsets=None, elem=None) -> None:
                self.offsets = offsets or []
                self.elem = elem

            # TODO needs to be supported by all iterators that represent tuples
            def __getitem__(self, index):
                return wrap_iterator(offsets=self.offsets, elem=index)

            def shift(self, *offsets):
                return wrap_iterator(offsets=[*self.offsets, *offsets], elem=self.elem)

            def max_neighbors(self):
                # TODO cleanup, test edge cases
                open_offsets = get_open_offsets(*self.offsets)
                assert open_offsets
                assert isinstance(
                    args[0].offset_provider[open_offsets[0].value],
                    NeighborTableOffsetProvider,
                )
                return args[0].offset_provider[open_offsets[0].value].max_neighbors

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


def named_range_(axis, range_):
    return ((axis, i) for i in range_)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(
            *map(lambda tup: named_range_(tup[0], tup[1]), domain.items())
        )
    )


def execute_shift(pos, tag, index, *, offset_provider):
    if tag in pos and pos[tag] is None:  # sparse field with offset as neighbor dimension
        new_pos = pos.copy()
        new_pos[tag] = index
        return new_pos
    assert tag.value in offset_provider
    offset_implementation = offset_provider[tag.value]
    if isinstance(offset_implementation, CartesianAxis):
        assert offset_implementation in pos
        new_pos = pos.copy()
        new_pos[offset_implementation] += index
        return new_pos
    elif isinstance(offset_implementation, NeighborTableOffsetProvider):
        assert offset_implementation.origin_axis in pos
        new_pos = pos.copy()
        del new_pos[offset_implementation.origin_axis]
        if offset_implementation.tbl[pos[offset_implementation.origin_axis], index] is None:
            return None
        else:
            new_pos[offset_implementation.neighbor_axis] = offset_implementation.tbl[
                pos[offset_implementation.origin_axis], index
            ]
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
def group_offsets(*offsets):
    tag_stack = []
    complete_offsets = []
    for offset in offsets:
        if not isinstance(offset, int):
            tag_stack.append(offset)
        else:
            assert tag_stack
            tag = tag_stack.pop(0)
            complete_offsets.append((tag, offset))
    return complete_offsets, tag_stack


def shift_position(pos, *complete_offsets, offset_provider):
    new_pos = pos.copy()
    for tag, index in complete_offsets:
        new_pos = execute_shift(new_pos, tag, index, offset_provider=offset_provider)
        if new_pos is None:
            return None
    return new_pos


def get_open_offsets(*offsets):
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


class MDIterator:
    def __init__(
        self, field, pos, *, incomplete_offsets=None, offset_provider, column_axis=None
    ) -> None:
        self.field = field
        self.pos = pos
        self.incomplete_offsets = incomplete_offsets or []
        self.offset_provider = offset_provider
        self.column_axis = column_axis

    def shift(self, *offsets):
        complete_offsets, open_offsets = group_offsets(*self.incomplete_offsets, *offsets)
        return MDIterator(
            self.field,
            shift_position(self.pos, *complete_offsets, offset_provider=self.offset_provider),
            incomplete_offsets=open_offsets,
            offset_provider=self.offset_provider,
            column_axis=self.column_axis,
        )

    def max_neighbors(self):
        assert self.incomplete_offsets
        assert isinstance(
            self.offset_provider[self.incomplete_offsets[0].value], NeighborTableOffsetProvider
        )
        return self.offset_provider[self.incomplete_offsets[0].value].max_neighbors

    def can_deref(self):
        return self.pos is not None

    def deref(self):
        if not self.can_deref():
            # this can legally happen in cases like `if_(can_deref(inp), deref(inp), 42.)`
            # because both branches will be eagerly executed
            return _UNDEFINED

        shifted_pos = self.pos.copy()
        # TODO(havogt): support nested tuples
        axises = self.field[0].axises if isinstance(self.field, tuple) else self.field.axises

        if not all(axis in shifted_pos.keys() for axis in axises):
            raise IndexError("Iterator position doesn't point to valid location for its field.")
        slice_column = {}
        if self.column_axis is not None:
            slice_column[self.column_axis] = slice(shifted_pos[self.column_axis], None)
            del shifted_pos[self.column_axis]
        ordered_indices = get_ordered_indices(
            axises,
            shifted_pos,
            slice_axises=slice_column,
        )
        try:
            if isinstance(self.field, tuple):
                return tuple(f[ordered_indices] for f in self.field)
            else:
                return self.field[ordered_indices]
        except IndexError:
            return _UNDEFINED


def make_in_iterator(inp, pos, offset_provider, *, column_axis):
    # TODO(havogt): support nested tuples
    axises = inp[0].axises if isinstance(inp, tuple) else inp.axises
    sparse_dimensions = [axis for axis in axises if isinstance(axis, Offset)]
    assert len(sparse_dimensions) <= 1  # TODO multiple is not a current use case
    new_pos = pos.copy()
    for axis in sparse_dimensions:
        new_pos[axis] = None
    if column_axis is not None:
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        new_pos[column_axis] = 0
    return MDIterator(
        inp,
        new_pos,
        incomplete_offsets=[*sparse_dimensions],
        offset_provider=offset_provider,
        column_axis=column_axis,
    )


builtins.builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


class LocatedField:
    """A Field with named dimensions/axises.

    Axis keys can be any objects that are hashable.
    """

    def __init__(self, getter, axises, dtype, *, setter=None, array=None):
        self.getter = getter
        self.axises = axises
        self.setter = setter
        self.array = array
        self.dtype = dtype

    def __getitem__(self, indices):
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices, value):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)

    def __array__(self):
        if self.array is None:
            raise TypeError("__array__ not supported for this field")
        return self.array()

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array().shape


def get_ordered_indices(axises, pos, *, slice_axises=None):
    """pos is a dictionary from axis to offset."""  # noqa: D403
    slice_axises = slice_axises or dict()
    assert all(axis in [*pos.keys(), *slice_axises] for axis in axises)
    return tuple(pos[axis] if axis in pos else slice_axises[axis] for axis in axises)


def _tupsum(a, b):
    def combine_slice(s, t):
        is_slice = False
        if isinstance(s, slice):
            is_slice = True
            first = s.start
            assert s.step is None
            assert s.stop is None
        else:
            assert isinstance(s, numbers.Integral)
            first = s
        if isinstance(t, slice):
            is_slice = True
            second = t.start
            assert t.step is None
            assert t.stop is None
        else:
            assert isinstance(t, numbers.Integral)
            second = t
        start = first + second
        return slice(start, None) if is_slice else start

    return tuple(combine_slice(*i) for i in zip(a, b))


def np_as_located_field(*axises, origin=None):
    def _maker(a: np.ndarray):
        if a.ndim != len(axises):
            raise TypeError("ndarray.ndim incompatible with number of given axises")

        if origin is not None:
            offsets = get_ordered_indices(axises, origin)
        else:
            offsets = tuple(0 for _ in axises)

        def setter(indices, value):
            indices = tupelize(indices)
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedField(getter, axises, dtype=a.dtype, setter=setter, array=a.__array__)

    return _maker


def index_field(axis, dtype=float):
    return LocatedField(lambda index: index[0], (axis,), dtype)


@builtins.shift.register(EMBEDDED)
def shift(*offsets):
    def impl(it):
        return it.shift(*offsets)

    return impl


@dataclass
class Column:
    axis: CartesianAxis
    range: range  # noqa: A003


class ScanArgIterator:
    def __init__(self, wrapped_iter, k_pos, *, offsets=None) -> None:
        self.wrapped_iter = wrapped_iter
        self.offsets = offsets or []
        self.k_pos = k_pos

    def deref(self):
        if not self.can_deref():
            return _UNDEFINED
        return self.wrapped_iter.deref()[self.k_pos]

    def can_deref(self):
        return self.wrapped_iter.can_deref()

    def shift(self, *offsets):
        return ScanArgIterator(self.wrapped_iter, offsets=[*offsets, *self.offsets])


def shifted_scan_arg(k_pos):
    def impl(it):
        return ScanArgIterator(it, k_pos=k_pos)

    return impl


def is_located_field(field) -> bool:
    return isinstance(field, LocatedField)  # TODO define on concept, not on concrete model


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
        return (field.axises,)


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
        if not all(axises == axeses[0] for axises in axeses):
            raise TypeError("All fields in the tuple need the same axises.")
        self.axises = axeses[0]

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


def fendef_embedded(fun, *args, **kwargs):  # noqa: 536
    if "offset_provider" not in kwargs:
        raise RuntimeError("offset_provider not provided")

    @iterator.runtime.closure.register(EMBEDDED)
    def closure(domain, sten, out, ins):  # domain is Dict[axis, range]
        if not (is_located_field(out) or can_be_tuple_field(out)):
            raise TypeError("Out needs to be a located field.")

        column = None
        if "column_axis" in kwargs:
            _column_axis = kwargs["column_axis"]
            column = Column(_column_axis, domain[_column_axis])
            del domain[_column_axis]

        @builtins.scan.register(
            EMBEDDED
        )  # TODO this is a bit ugly, alternative: pass scan range via iterator
        def scan(scan_pass, is_forward, init):
            def impl(*iters):
                if column is None:
                    raise RuntimeError("Column axis is not defined, cannot scan.")

                _range = column.range
                if not is_forward:
                    _range = reversed(_range)

                state = init
                col = []
                for i in _range:
                    state = scan_pass(
                        state, *map(shifted_scan_arg(i), iters)
                    )  # more generic scan returns state and result as 2 different things
                    col.append(state)

                if not is_forward:
                    col = np.flip(col)

                if isinstance(col[0], tuple):
                    dtype = ", ".join(np.dtype(type(c)).str for c in col[0])
                    return np.asarray(col, dtype=dtype)

                return np.asarray(col)

            return impl

        out = as_tuple_field(out) if can_be_tuple_field(out) else out

        for pos in domain_iterator(domain):
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
                ordered_indices = get_ordered_indices(out.axises, pos)
                out[ordered_indices] = res
            else:
                colpos = pos.copy()
                for k in column.range:
                    colpos[column.axis] = k
                    ordered_indices = get_ordered_indices(out.axises, colpos)
                    out[ordered_indices] = res[k]

    fun(*args)


iterator.runtime.fendef_embedded = fendef_embedded
