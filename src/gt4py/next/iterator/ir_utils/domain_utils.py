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
import warnings
from typing import Callable, Iterable, Literal, Optional

import numpy as np

from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
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

    def __post_init__(self) -> None:
        # TODO(havogt): added this defensive checks as code seems to make this reasonable assumption
        assert self.start is not itir.InfinityLiteral.POSITIVE
        assert self.stop is not itir.InfinityLiteral.NEGATIVE

    def translate(self, distance: int) -> SymbolicRange:
        return SymbolicRange(im.plus(self.start, distance), im.plus(self.stop, distance))

    def empty(self) -> bool | None:
        # constant fold so that translated bounds like `0 + 1` are recognized as literals
        start = ConstantFolding.apply(self.start)
        stop = ConstantFolding.apply(self.stop)
        if isinstance(start, itir.Literal) and isinstance(stop, itir.Literal):
            return int(start.value) >= int(stop.value)
        elif start == stop:
            return True
        return None


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
                # cartesian shift: extract the (co)domain directly from the `CartesianOffset`
                if val is trace_shifts.Sentinel.VALUE:
                    raise NotImplementedError("Dynamic cartesian offsets not supported.")
                assert isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)

                old_dim = common.Dimension(value=off.domain.value, kind=off.domain.kind)
                new_dim = common.Dimension(value=off.codomain.value, kind=off.codomain.kind)

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
                        val,  # type: ignore[arg-type] # mypy not smart enough
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
    """
    Uses start_op and stop_op to fold the start and stop of a list of ranges.

    This function only computes the correct value if the (non-empty) ranges are either
    overlapping / adjacent or empty, as calculation is by means of the convex hull. Non-static
    ranges may not be empty for now.
    """
    non_empty_ranges = [range_ for range_ in ranges if not range_.empty()]
    if len(non_empty_ranges) == 0:
        return ranges[0]  # all empty -> empty
    elif len(non_empty_ranges) == 1:
        # the reduction of a single range is the range itself
        return non_empty_ranges[0]
    else:
        start = functools.reduce(start_reduce_op, [range_.start for range_ in non_empty_ranges])
        stop = functools.reduce(stop_reduce_op, [range_.stop for range_ in non_empty_ranges])
    # constant fold to keep the tree small and so translated bounds (e.g. `0 + 1`) collapse to a
    # literal (we deliberately do not fold in `translate`)
    start, stop = ConstantFolding.apply(start), ConstantFolding.apply(stop)  # type: ignore[assignment]  # always an itir.Expr
    return SymbolicRange(start, stop)


_range_union = functools.partial(
    _reduce_ranges, start_reduce_op=im.minimum, stop_reduce_op=im.maximum
)
_range_intersection = functools.partial(
    _reduce_ranges, start_reduce_op=im.maximum, stop_reduce_op=im.minimum
)


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
