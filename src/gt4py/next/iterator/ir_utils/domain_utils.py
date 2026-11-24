# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import itertools
import warnings
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np

from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im, misc
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


#: Threshold fraction of domain points which may be added to a domain on translation in order
#: to have a contiguous domain before a warning is raised.
_NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD: float = 1 / 4

#: Offset tags for which a non-contiguous domain warning has already been printed
_NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS: set[str] = set()


@dataclasses.dataclass(frozen=True)
class SymbolicRange:
    start: itir.Expr
    stop: itir.Expr
    #: The parts the range is the union of, if one of them may be empty at runtime, else `()`.
    #: `start` and `stop` are the bounds of the union of the parts, see `_union_of_parts`. The parts
    #: describe how the range was derived and allow to compute unions and intersections with
    #: other ranges without taking the convex hull with a part that is empty at runtime. They are
    #: not part of the value of the range and hence excluded from comparison and hashing.
    parts: tuple[_RangePart, ...] = dataclasses.field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        # TODO(havogt): added this defensive checks as code seems to make this reasonable assumption
        assert self.start is not itir.InfinityLiteral.POSITIVE
        assert self.stop is not itir.InfinityLiteral.NEGATIVE

    def translate(self, distance: int) -> SymbolicRange:
        return SymbolicRange(
            im.plus(self.start, distance),
            im.plus(self.stop, distance),
            tuple(part.translate(distance) for part in self.parts),
        )

    def empty(self) -> bool | None:
        # constant fold so that translated bounds like `0 + 1` are recognized as literals
        start = ConstantFolding.apply(self.start)
        stop = ConstantFolding.apply(self.stop)
        if isinstance(start, itir.Literal) and isinstance(stop, itir.Literal):
            return int(start.value) >= int(stop.value)
        elif start == stop:
            return True
        return None


@dataclasses.dataclass(frozen=True)
class _RangePart:
    """
    A part of a `SymbolicRange`, which may be empty, although this is not known statically.

    Ranges are assumed to be non-empty, unless they are statically empty (see
    `SymbolicRange.empty`). The intersection of non-empty ranges may however be empty, e.g.,
    `[0, n[ & [c, inf[` for `c >= n`, which happens for the regions selected by the branches of a
    `concat_where` if the selected region is empty at runtime. Such a part carries the condition
    `empty_if` under which it is empty (`start >= stop`) and the `envelope`, a range assumed to be
    non-empty which contains the part, i.e., the range the part was cut out of.
    """

    range: SymbolicRange
    #: A condition evaluated at runtime under which `range` is empty, `None` if the part is assumed
    #: to be non-empty.
    empty_if: Optional[itir.Expr]
    envelope: SymbolicRange

    def translate(self, distance: int) -> _RangePart:
        return _RangePart(
            self.range.translate(distance), self.empty_if, self.envelope.translate(distance)
        )


_GRID_TYPE_MAPPING = {
    "unstructured_domain": common.GridType.UNSTRUCTURED,
    "cartesian_domain": common.GridType.CARTESIAN,
}


def _unstructured_translate_range_statically(
    range_: SymbolicRange,
    tag: str,
    val: itir.OffsetLiteral
    | Literal[trace_shifts.Sentinel.VALUE, trace_shifts.Sentinel.ALL_NEIGHBORS],
    connectivity: common.NeighborTable,
    expr: itir.Expr | None = None,
) -> SymbolicRange:
    """
    Translate `range_` using static connectivity information from `offset_provider`.
    """
    assert common.is_neighbor_table(connectivity)
    skip_value = connectivity.skip_value

    # fold & convert expr into actual integers
    start_expr, stop_expr = range_.start, range_.stop
    # note: if you find tuple expressions on literals here, you likely forgot to collapse tuple
    # expressions beforehand
    assert isinstance(start_expr, itir.Literal) and isinstance(stop_expr, itir.Literal)
    start, stop = int(start_expr.value), int(stop_expr.value)

    if range_.empty():
        return SymbolicRange(
            im.literal(str("0"), builtins.INTEGER_INDEX_BUILTIN),
            im.literal(str("0"), builtins.INTEGER_INDEX_BUILTIN),
        )

    nb_index: slice | int
    if val in [trace_shifts.Sentinel.ALL_NEIGHBORS, trace_shifts.Sentinel.VALUE]:
        nb_index = slice(None)
    else:
        nb_index = val.value  # type: ignore[assignment]  # assert above

    accessed = connectivity.ndarray[start:stop, nb_index]

    if isinstance(val, itir.OffsetLiteral) and np.any(accessed == skip_value):
        # TODO(tehrengruber): Turn this into a configurable error. This is currently
        #  not possible since some test cases starting from ITIR containing
        #  `can_deref` might lead here. The frontend never emits such IR and domain
        #  inference runs after we transform reductions into stmts containing
        #  `can_deref`.
        warnings.warn(
            UserWarning(f"Translating '{expr}' using '{tag}' has an out-of-bounds access."),
            stacklevel=2,
        )

    new_start, new_stop = accessed.min(), accessed.max() + 1  # type: ignore[attr-defined]  # TODO(havogt): improve typing for NDArrayObject

    fraction_accessed = np.unique(accessed).size / (new_stop - new_start)  # type: ignore[call-overload]  # TODO(havogt): improve typing for NDArrayObject

    if fraction_accessed < _NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD and (
        tag not in _NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS
    ):
        _NON_CONTIGUOUS_DOMAIN_WARNING_SKIPPED_OFFSET_TAGS.add(tag)
        warnings.warn(
            UserWarning(
                f"Translating '{expr}' using '{tag}' requires "
                f"computations on many additional points "
                f"({round((1 - fraction_accessed) * 100)}%) in order to get a contiguous "
                f"domain. Please consider reordering your mesh."
            ),
            stacklevel=2,
        )

    return SymbolicRange(
        im.literal(str(new_start), builtins.INTEGER_INDEX_BUILTIN),
        im.literal(str(new_stop), builtins.INTEGER_INDEX_BUILTIN),
    )


@dataclasses.dataclass(frozen=True)
class SymbolicDomain:
    grid_type: common.GridType
    ranges: dict[common.Dimension, SymbolicRange]

    def __hash__(self) -> int:
        return hash((self.grid_type, frozenset(self.ranges.items())))

    def empty(self) -> bool | None:
        if any(r.empty() for r in self.ranges.values()):
            return True
        if any(r.empty() is None for r in self.ranges.values()):
            return None
        return False

    @classmethod
    def from_expr(cls, node: itir.Node) -> SymbolicDomain:
        assert cpm.is_call_to(node, ("unstructured_domain", "cartesian_domain"))

        ranges: dict[common.Dimension, SymbolicRange] = {}
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, itir.AxisLiteral)

            ranges[common.Dimension(value=axis_literal.value, kind=axis_literal.kind)] = (
                SymbolicRange(lower_bound, upper_bound)
            )
        return cls(_GRID_TYPE_MAPPING[node.fun.id], ranges)

    def as_expr(self) -> itir.FunCall:
        converted_ranges: dict[common.Dimension, tuple[itir.Expr, itir.Expr]] = {
            key: (value.start, value.stop) for key, value in self.ranges.items()
        }
        return im.domain(self.grid_type, converted_ranges)

    def translate(
        self: SymbolicDomain,
        shift: tuple[
            itir.OffsetLiteral
            | itir.CartesianOffset
            | Literal[trace_shifts.Sentinel.VALUE, trace_shifts.Sentinel.ALL_NEIGHBORS],
            ...,
        ],
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        #: A dictionary mapping axes names to their length. See
        #: func:`gt4py.next.iterator.transforms.infer_domain.infer_expr` for more details.
        symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    ) -> SymbolicDomain:
        dims = list(self.ranges.keys())
        new_ranges = {dim: self.ranges[dim] for dim in dims}
        if len(shift) == 0:
            return self
        if len(shift) == 2:
            off, val = shift

            if isinstance(off, itir.CartesianOffset):
                if val is trace_shifts.Sentinel.VALUE:
                    raise NotImplementedError("Dynamic cartesian offsets not supported.")
                assert isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)

                old_dim = misc.dim_from_axis_literal(off.domain)
                new_dim = misc.dim_from_axis_literal(off.codomain)

                assert new_dim not in new_ranges or old_dim == new_dim

                new_range = SymbolicRange.translate(self.ranges[old_dim], val.value)
                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )
            else:
                assert isinstance(off, itir.OffsetLiteral) and isinstance(off.value, str)
                assert (
                    isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                ) or val in [
                    trace_shifts.Sentinel.ALL_NEIGHBORS,
                    trace_shifts.Sentinel.VALUE,
                ]

                connectivity: common.NeighborTable | common.NeighborConnectivityType
                if common.is_offset_provider(offset_provider):
                    connectivity = common.get_offset(offset_provider, off.value)
                    old_dim = connectivity.domain.dims[0]
                    new_dim = connectivity.codomain
                else:
                    assert common.is_offset_provider_type(offset_provider)
                    connectivity = common.get_offset_type(offset_provider, off.value)
                    old_dim = connectivity.domain[0]
                    new_dim = connectivity.codomain

                assert new_dim not in new_ranges or old_dim == new_dim
                if symbolic_domain_sizes is not None and new_dim.value in symbolic_domain_sizes:
                    new_range = SymbolicRange(
                        im.literal(str(0), builtins.INTEGER_INDEX_BUILTIN),
                        im.ensure_expr(symbolic_domain_sizes[new_dim.value]),
                    )
                else:
                    assert common.is_neighbor_table(connectivity)
                    new_range = _unstructured_translate_range_statically(
                        new_ranges[old_dim],
                        off.value,
                        val,
                        connectivity,
                        self.as_expr(),
                    )

                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )

            return SymbolicDomain(self.grid_type, new_ranges)
        elif len(shift) > 2:
            return self.translate(shift[0:2], offset_provider, symbolic_domain_sizes).translate(
                shift[2:], offset_provider, symbolic_domain_sizes
            )
        else:
            raise AssertionError("Number of shifts must be a multiple of 2.")


def _reduce_ranges(
    *ranges: SymbolicRange,
    start_reduce_op: Callable[[itir.Expr, itir.Expr], itir.Expr],
    stop_reduce_op: Callable[[itir.Expr, itir.Expr], itir.Expr],
) -> SymbolicRange:
    """Uses start_op and stop_op to fold the start and stop of a list of ranges."""
    start = functools.reduce(start_reduce_op, [range_.start for range_ in ranges])
    stop = functools.reduce(stop_reduce_op, [range_.stop for range_ in ranges])
    # constant fold to keep the tree small and so translated bounds (e.g. `0 + 1`) collapse to a
    # literal (we deliberately do not fold in `translate`)
    start, stop = ConstantFolding.apply(start), ConstantFolding.apply(stop)  # type: ignore[assignment]  # always an itir.Expr
    return SymbolicRange(start, stop)


#: A linear expression `sum(coefficient * symbol) + constant` as `(coefficients, constant)`.
_LinearForm = tuple[frozenset[tuple[str, int]], int]


def _linear_form(expr: itir.Expr) -> Optional[_LinearForm]:
    """
    Return `expr` in linear form if it is a sum of symbols and integer literals.

    >>> coefficients, constant = _linear_form(im.minus(im.plus("n", 1), "m"))
    >>> sorted(coefficients), constant
    ([('m', -1), ('n', 1)], 1)
    """
    if isinstance(expr, itir.Literal):
        return (frozenset(), int(expr.value)) if expr.value.lstrip("-").isdigit() else None
    if isinstance(expr, itir.SymRef):
        return frozenset({(str(expr.id), 1)}), 0
    if cpm.is_call_to(expr, ("plus", "minus")):
        lhs, rhs = (_linear_form(arg) for arg in expr.args)
        if lhs is None or rhs is None:
            return None
        if cpm.is_call_to(expr, "plus"):
            rhs = _subtract((frozenset(), 0), rhs)
        return _subtract(lhs, rhs)
    return None


def _subtract(a: _LinearForm, b: _LinearForm) -> _LinearForm:
    coefficients = dict(a[0])
    for symbol, coefficient in b[0]:
        coefficients[symbol] = coefficients.get(symbol, 0) - coefficient
    return frozenset((s, c) for s, c in coefficients.items() if c != 0), a[1] - b[1]


def _linear_difference(condition: itir.Expr) -> Optional[_LinearForm]:
    """Return `lhs - rhs` in linear form for a condition `lhs >= rhs`."""
    if not cpm.is_call_to(condition, "greater_equal"):
        return None
    lhs, rhs = (_linear_form(arg) for arg in condition.args)
    return None if lhs is None or rhs is None else _subtract(lhs, rhs)


def _implied(fact: _LinearForm, facts: Sequence[_LinearForm]) -> bool:
    """
    Return whether `fact >= 0` follows from `facts >= 0`.

    This is the case if `fact` is the sum of (at most two of) the `facts` and of a non-negative
    constant. The check is sufficient, not necessary.
    """
    for num_facts in range(min(len(facts), 2) + 1):
        for combination in itertools.combinations(facts, num_facts):
            remainder = functools.reduce(_subtract, combination, fact)
            if not remainder[0] and remainder[1] >= 0:
                return True
    return False


def _flat_args(expr: itir.Expr, op: str) -> list[itir.Expr]:
    """
    Return the arguments of the (nested) calls to the associative builtin `op` in `expr`.

    >>> [str(arg) for arg in _flat_args(im.maximum(im.maximum("a", "b"), "c"), "maximum")]
    ['a', 'b', 'c']
    """
    if not cpm.is_call_to(expr, op):
        return [expr]
    return [flat_arg for arg in expr.args for flat_arg in _flat_args(arg, op)]


def _push_offset(expr: itir.Expr) -> itir.Expr:
    """`minimum(a, b) + c` -> `minimum(a + c, b + c)`, also for `maximum` and `minus`."""
    if cpm.is_call_to(expr, ("plus", "minus")):
        lhs, offset = expr.args
        lhs = _push_offset(lhs)
        if cpm.is_call_to(lhs, ("minimum", "maximum")) and _linear_form(offset) is not None:
            return im.call(lhs.fun.id)(
                *(_push_offset(im.call(expr.fun.id)(arg, offset)) for arg in lhs.args)
            )
    return expr


def _provably_less_equal(a: itir.Expr, b: itir.Expr, facts: Sequence[_LinearForm] = ()) -> bool:
    """Return whether `a <= b` follows from `facts >= 0`. Sufficient, not necessary."""
    a, b = _push_offset(a), _push_offset(b)
    if cpm.is_call_to(b, "minimum"):
        return all(_provably_less_equal(a, arg, facts) for arg in b.args)
    if cpm.is_call_to(a, "maximum"):
        return all(_provably_less_equal(arg, b, facts) for arg in a.args)
    if cpm.is_call_to(a, "minimum"):
        return any(_provably_less_equal(arg, b, facts) for arg in a.args)
    if cpm.is_call_to(b, "maximum"):
        return any(_provably_less_equal(a, arg, facts) for arg in b.args)
    a_form, b_form = _linear_form(a), _linear_form(b)
    if a_form is None or b_form is None:
        return a == b
    return _implied(_subtract(b_form, a_form), facts)


def _simplify_min_max(expr: itir.Expr, op: Literal["minimum", "maximum"]) -> itir.Expr:
    """
    Drop the arguments of a (nested) call to `op` that are never its result.

    >>> str(_simplify_min_max(im.maximum(im.maximum("n", 0), im.plus("n", 1)), "maximum"))
    'maximum(0, n + 1)'
    """

    def dominated(a: itir.Expr, b: itir.Expr) -> bool:
        # `a` is never the result of `maximum(a, b)` if `a <= b`
        return _provably_less_equal(a, b) if op == "maximum" else _provably_less_equal(b, a)

    kept: list[itir.Expr] = []
    for arg in _flat_args(_push_offset(expr), op):
        if not any(dominated(arg, other) for other in kept):
            kept = [other for other in kept if not dominated(other, arg)] + [arg]
    return functools.reduce(im.call(op), kept)


#: A disjunction of conditions, see `_add_condition`.
_Disjunction = dict[itir.Expr | frozenset[tuple[str, int]], tuple[int, itir.Expr]]


def _add_condition(disjunction: _Disjunction, condition: itir.Expr) -> None:
    """
    Add a condition to a disjunction of conditions, dropping the ones implied by another one.

    `a + k1 >= b` implies `a + k2 >= b` for `k1 <= k2`, so only the latter is kept, e.g.,
    `n >= m | n + 1 >= m` is `n + 1 >= m`.
    """
    if cpm.is_call_to(condition, "or_"):
        for arg in condition.args:
            _add_condition(disjunction, arg)
        return
    key: itir.Expr | frozenset[tuple[str, int]] = condition
    weakness = 0
    if (difference := _linear_difference(condition)) is not None:
        # `lhs >= rhs` <=> `lhs - rhs >= 0`, the larger the constant the weaker the condition
        key, weakness = difference
    if key not in disjunction or disjunction[key][0] < weakness:
        disjunction[key] = (weakness, condition)


def _range_hull(*ranges: SymbolicRange, simplify: bool = False) -> SymbolicRange:
    hull = _reduce_ranges(*ranges, start_reduce_op=im.minimum, stop_reduce_op=im.maximum)
    if not simplify:
        return hull
    return SymbolicRange(
        _simplify_min_max(hull.start, "minimum"), _simplify_min_max(hull.stop, "maximum")
    )


def _parts(range_: SymbolicRange) -> tuple[_RangePart, ...]:
    """Return the parts of a range; a range without parts is its only, non-empty, part."""
    return range_.parts or (_RangePart(range_, None, range_),)


def _intersection_of_parts(parts: Sequence[_RangePart]) -> Optional[_RangePart]:
    """Return the intersection of `parts`, or `None` if it is statically empty."""
    range_ = _reduce_ranges(
        *(part.range for part in parts), start_reduce_op=im.maximum, stop_reduce_op=im.minimum
    )
    range_ = SymbolicRange(
        _simplify_min_max(range_.start, "maximum"), _simplify_min_max(range_.stop, "minimum")
    )
    if range_.empty():
        return None
    # The intersection is empty if one of the parts is empty or if two of them do not overlap,
    # i.e., `maximum(a, ...) >= minimum(b, ...)`, which holds if `a >= b` for any of the bounds
    # composing the start of one and the stop of the other part. Comparing these instead of the
    # starts and stops allows to detect more statically empty intersections and keeps the
    # condition small. For the intersection of a range with a `concat_where` condition, i.e., a
    # half-infinite range, this is a single comparison.
    disjunction: _Disjunction = {}
    for part in parts:
        if part.empty_if is not None:
            _add_condition(disjunction, part.empty_if)
    for part_a, part_b in itertools.permutations(parts, 2):
        for start in _flat_args(_push_offset(part_a.range.start), "maximum"):
            for stop in _flat_args(_push_offset(part_b.range.stop), "minimum"):
                if start == itir.InfinityLiteral.NEGATIVE or stop == itir.InfinityLiteral.POSITIVE:
                    continue
                no_overlap: itir.Expr = ConstantFolding.apply(im.greater_equal(start, stop))  # type: ignore[assignment]  # always an itir.Expr
                difference = _linear_difference(no_overlap)
                if difference is not None and not difference[0]:  # e.g. `n + 1 >= n`
                    no_overlap = im.literal_from_value(difference[1] >= 0)
                if no_overlap == im.literal_from_value(True):
                    return None
                if no_overlap != im.literal_from_value(False):
                    _add_condition(disjunction, no_overlap)
    envelopes = [part.envelope for part in parts if is_finite(part.envelope)]
    if not disjunction or not envelopes:
        # never empty, or not bounded such that the bounds can be guarded in `_union_of_parts`:
        # assume the intersection to be non-empty (over-approximation)
        return _RangePart(range_, None, range_)
    empty_if = functools.reduce(im.or_, (condition for _, condition in disjunction.values()))
    return _RangePart(range_, empty_if, envelopes[0])


def _needs_guard(
    part: _RangePart, others: Sequence[_RangePart], side: Literal["start", "stop"]
) -> bool:
    """
    Return whether the `side` bound of `part` must be guarded in the union with `others`.

    The guard is not needed if whenever `part.empty_if` holds (for each of its disjuncts), the
    bound of each other part that is non-empty dominates the bound of `part`, which then does not
    change the union, unless all parts are empty. This is the case, e.g., for the two regions
    selected by the branches of `concat_where(I < c, ...)`: if one is empty, the other one is
    the whole domain. In contrast, the two regions selected by the false branch of
    `concat_where(I == c, ...)` leave a gap at `c`, which the convex hull fills if one of them is
    empty. The envelopes are assumed to be non-empty.
    """
    assert part.empty_if is not None

    def disjuncts(condition: itir.Expr) -> Optional[list[_LinearForm]]:
        differences = [_linear_difference(arg) for arg in _flat_args(condition, "or_")]
        return None if None in differences else differences  # type: ignore[return-value]  # checked

    def non_empty(range_: SymbolicRange) -> list[_LinearForm]:
        start, stop = _linear_form(range_.start), _linear_form(range_.stop)
        if start is None or stop is None:
            return []
        return [_subtract(_subtract(stop, start), (frozenset(), 1))]  # `stop - start - 1 >= 0`

    part_disjuncts = disjuncts(part.empty_if)
    if part_disjuncts is None:
        return True
    for disjunct in part_disjuncts:
        for other in others:
            facts = [disjunct, *non_empty(part.envelope), *non_empty(other.envelope)]
            # `other` is non-empty, i.e., none of the disjuncts `lhs - rhs >= 0` of its condition
            # holds: `rhs - lhs - 1 >= 0`
            other_disjuncts = [] if other.empty_if is None else disjuncts(other.empty_if) or []
            facts += [_subtract((frozenset(), -1), d) for d in other_disjuncts]
            if side == "start":
                dominated = _provably_less_equal(other.range.start, part.range.start, facts)
            else:
                dominated = _provably_less_equal(part.range.stop, other.range.stop, facts)
            if not dominated:
                return True
    return False


def _union_of_parts(parts: Sequence[_RangePart]) -> SymbolicRange:
    """
    Return the convex hull of the parts that are non-empty at runtime.

    Parts with the same `empty_if` condition, e.g., the translations of one part accessed with
    several shifts, are empty at the same time and are combined into their convex hull first,
    and so are all parts that are assumed to be non-empty. If a single part remains, it is the
    result, without any runtime check. Otherwise, the bounds of the parts that may be empty are
    guarded by their condition, unless this is not needed (see `_needs_guard`): an empty part
    contributes the stop of the convex hull of all envelopes to the start of the union, which is
    not smaller than the start of any non-empty part (each of which is contained in its
    envelope), and vice versa for the stop. This keeps the size of the expressions linear in the
    number of parts.

    The result is exact (the convex hull of the non-empty parts) if at least one part is
    non-empty. If all parts are empty, i.e., the range is not needed at all, the result may be
    non-empty, like the convex hull computed without taking emptiness into account.
    """
    assert parts
    groups: dict[Optional[itir.Expr], list[_RangePart]] = {}
    for part in parts:
        groups.setdefault(part.empty_if, []).append(part)
    merged = [
        _RangePart(
            _range_hull(*(part.range for part in group), simplify=True),
            empty_if,
            _range_hull(*(part.envelope for part in group), simplify=True),
        )
        for empty_if, group in groups.items()
    ]
    if None in groups:
        # a part contained in a range which is part of the union anyway does not contribute
        (non_empty_part,) = [part for part in merged if part.empty_if is None]
        merged = [
            part
            for part in merged
            if part.empty_if is None or part.envelope != non_empty_part.range
        ]

    if len(merged) == 1:
        (part,) = merged
        return dataclasses.replace(part.range, parts=() if part.empty_if is None else (part,))

    envelope = _range_hull(*(part.envelope for part in merged))
    starts, stops = [], []
    for i, part in enumerate(merged):
        others = merged[:i] + merged[i + 1 :]
        start, stop = part.range.start, part.range.stop
        if part.empty_if is not None and _needs_guard(part, others, "start"):
            start = im.if_(part.empty_if, envelope.stop, start)
        if part.empty_if is not None and _needs_guard(part, others, "stop"):
            stop = im.if_(part.empty_if, envelope.start, stop)
        starts.append(start)
        stops.append(stop)
    return SymbolicRange(
        ConstantFolding.apply(functools.reduce(im.minimum, starts)),  # type: ignore[arg-type]  # always an itir.Expr
        ConstantFolding.apply(functools.reduce(im.maximum, stops)),  # type: ignore[arg-type]  # always an itir.Expr
        tuple(merged),
    )


def _range_union(*ranges: SymbolicRange) -> SymbolicRange:
    """
    Return the union of a list of ranges, computed as the convex hull.

    This only computes the correct value if the non-empty ranges are overlapping or adjacent.
    An empty range is the identity element of the union, so empty ranges are dropped rather than
    fed into the convex hull, which would otherwise over-approximate (e.g. `[10, 10[ | [11, 11[`
    is empty, but its convex hull is `[10, 11[`). Emptiness is only decidable for static ranges.
    A part of a range that may be empty at runtime (see `_RangePart`) only contributes to the
    convex hull if it is non-empty, which is decided at runtime (see `_union_of_parts`). All other
    non-static ranges are assumed to be non-empty.
    """
    non_empty_ranges = [range_ for range_ in ranges if not range_.empty()]
    if not non_empty_ranges:
        return ranges[0]  # all empty -> empty
    if not any(range_.parts for range_ in non_empty_ranges):
        return _range_hull(*non_empty_ranges)
    return _union_of_parts(
        [part for range_ in non_empty_ranges for part in _parts(range_) if not part.range.empty()]
    )


def _range_intersection(*ranges: SymbolicRange) -> SymbolicRange:
    """
    Return the intersection of a list of ranges.

    An empty range is the absorbing element of the intersection, so it is returned directly.
    Folding it into `max`/`min` instead would give a wrong (possibly non-empty, possibly
    unbounded) result, e.g. `[1, 1[ & [0, inf[` would yield `[1, inf[`. Emptiness is only
    decidable for static ranges. The intersection of non-empty ranges may however be empty at
    runtime, e.g. `[0, n[ & [c, inf[` for `c >= n`. The result then records the condition under
    which this is the case (see `_RangePart`). The intersection with a range composed of parts
    is the union of the intersections with each part.
    """
    for range_ in ranges:
        if range_.empty():
            return range_
    result = _reduce_ranges(*ranges, start_reduce_op=im.maximum, stop_reduce_op=im.minimum)
    # the universe `]-inf, inf[` is the identity element of the intersection
    bounded_ranges = [
        range_
        for range_ in ranges
        if (range_.start, range_.stop)
        != (itir.InfinityLiteral.NEGATIVE, itir.InfinityLiteral.POSITIVE)
    ]
    if len(bounded_ranges) <= 1:
        return dataclasses.replace(result, parts=bounded_ranges[0].parts if bounded_ranges else ())
    parts = [
        part
        for combination in itertools.product(*(_parts(range_) for range_ in bounded_ranges))
        if (part := _intersection_of_parts(combination)) is not None
    ]
    if not parts:  # statically empty
        if result.empty():
            return result
        zero = im.literal("0", builtins.INTEGER_INDEX_BUILTIN)
        return SymbolicRange(zero, zero)
    if not any(range_.parts for range_ in bounded_ranges):
        # intersection of ranges without parts: keep the bounds as they are
        (part,) = parts
        if part.empty_if is None:
            return result
        return dataclasses.replace(result, parts=(dataclasses.replace(part, range=result),))
    return _union_of_parts(parts)


def _reduce_domains(
    *domains: SymbolicDomain,
    range_reduce_op: Callable[..., SymbolicRange],
) -> SymbolicDomain:
    """
    Applies range_op to the ranges of a list of domains with same dimensions and grid_type.
    """
    assert all(domain.grid_type == domains[0].grid_type for domain in domains)
    assert all(domain.ranges.keys() == domains[0].ranges.keys() for domain in domains)

    dims = domains[0].ranges.keys()
    new_domain_ranges = {dim: range_reduce_op(*(d.ranges[dim] for d in domains)) for dim in dims}

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)


domain_union = functools.partial(_reduce_domains, range_reduce_op=_range_union)
"""Return the (set) union of a list of domains."""

domain_intersection = functools.partial(_reduce_domains, range_reduce_op=_range_intersection)
"""Return the intersection of a list of domains."""


def domain_complement(domain: SymbolicDomain) -> SymbolicDomain:
    """
    Return the (set) complement of a half-infinite domain.

    Note: after canonicalization of concat_where, the domain is always half-infinite,
    i.e. it has ranges of the form `]-inf, a[` or `[a, inf[`.
    """
    dims_dict = {}
    for dim in domain.ranges.keys():
        lb, ub = domain.ranges[dim].start, domain.ranges[dim].stop
        assert (lb == itir.InfinityLiteral.NEGATIVE) != (ub == itir.InfinityLiteral.POSITIVE)
        # `]-inf, a[` -> `[a, inf[`
        if lb == itir.InfinityLiteral.NEGATIVE:
            dims_dict[dim] = SymbolicRange(start=ub, stop=itir.InfinityLiteral.POSITIVE)
        # `[a, inf]` -> `]-inf, a]`
        else:  # ub == itir.InfinityLiteral.POSITIVE:
            dims_dict[dim] = SymbolicRange(start=itir.InfinityLiteral.NEGATIVE, stop=lb)
    return SymbolicDomain(domain.grid_type, dims_dict)


def promote_domain(
    domain: SymbolicDomain, target_dims: Iterable[common.Dimension]
) -> SymbolicDomain:
    """Return a domain that is extended with the dimensions of target_dims."""
    assert set(domain.ranges.keys()).issubset(target_dims)
    dims_dict = {
        dim: domain.ranges[dim]
        if dim in domain.ranges
        else SymbolicRange(itir.InfinityLiteral.NEGATIVE, itir.InfinityLiteral.POSITIVE)
        for dim in target_dims
    }
    return SymbolicDomain(domain.grid_type, dims_dict)


def is_finite(range_or_domain: SymbolicRange | SymbolicDomain) -> bool:
    """
    Return whether a range is unbounded in (at least) one direction.

    The expression is required to be constant folded before for the result to be reliable.
    """
    match range_or_domain:
        case SymbolicRange() as range_:
            infinity_literals = (itir.InfinityLiteral.POSITIVE, itir.InfinityLiteral.NEGATIVE)
            return not (range_.start in infinity_literals or range_.stop in infinity_literals)
        case SymbolicDomain() as domain:
            return all(is_finite(range_) for range_ in domain.ranges.values())
        case _:
            raise ValueError("Expected a 'SymbolicRange' or 'SymbolicDomain'.")
