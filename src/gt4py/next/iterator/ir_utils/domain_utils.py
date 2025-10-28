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
from gt4py.next.iterator.transforms import collapse_tuple, trace_shifts
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


#: Threshold fraction of domain points which may be added to a domain on translation in order
#: to have a contiguous domain before a warning is raised.
NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD: float = 1 / 4

#: Skip printing warnings after exceeding `NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD` this many times.
NON_CONTIGUOUS_DOMAIN_MAX_WARNINGS: int = 5

#: Number of warnings raised after exceeding `NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD`
_NON_CONTIGUOUS_DOMAIN_WARNING_COUNTER: int = 0


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
        if isinstance(self.start, itir.Literal) and isinstance(self.stop, itir.Literal):
            start, stop = int(self.start.value), int(self.stop.value)
            return start >= stop
        elif self.start == self.stop:
            return True
        return None


_GRID_TYPE_MAPPING = {
    "unstructured_domain": common.GridType.UNSTRUCTURED,
    "cartesian_domain": common.GridType.CARTESIAN,
}


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
            | Literal[trace_shifts.Sentinel.VALUE, trace_shifts.Sentinel.ALL_NEIGHBORS],
            ...,
        ],
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        #: A dictionary mapping axes names to their length. See
        #: func:`gt4py.next.iterator.transforms.infer_domain.infer_expr` for more details.
        symbolic_domain_sizes: Optional[dict[str, str | itir.Expr]] = None,
    ) -> SymbolicDomain:
        offset_provider_type = common.offset_provider_to_type(offset_provider)

        dims = list(self.ranges.keys())
        new_ranges = {dim: self.ranges[dim] for dim in dims}
        if len(shift) == 0:
            return self
        if len(shift) == 2:
            off, val = shift
            assert isinstance(off, itir.OffsetLiteral) and isinstance(off.value, str)
            connectivity_type = common.get_offset_type(offset_provider_type, off.value)

            if isinstance(connectivity_type, common.Dimension):
                if val is trace_shifts.Sentinel.VALUE:
                    raise NotImplementedError("Dynamic offsets not supported.")
                assert isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                current_dim = connectivity_type
                # cartesian offset
                new_ranges[current_dim] = SymbolicRange.translate(
                    self.ranges[current_dim], val.value
                )
            elif isinstance(connectivity_type, common.NeighborConnectivityType):
                # unstructured shift
                assert (
                    isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                ) or val in [
                    trace_shifts.Sentinel.ALL_NEIGHBORS,
                    trace_shifts.Sentinel.VALUE,
                ]
                old_dim = connectivity_type.source_dim
                new_dim = connectivity_type.codomain
                assert new_dim not in new_ranges or old_dim == new_dim
                if symbolic_domain_sizes is not None and new_dim.value in symbolic_domain_sizes:
                    new_range = SymbolicRange(
                        im.literal(str(0), builtins.INTEGER_INDEX_BUILTIN),
                        im.ensure_expr(symbolic_domain_sizes[new_dim.value]),
                    )
                else:
                    assert common.is_offset_provider(offset_provider)
                    connectivity = offset_provider[off.value]
                    assert isinstance(connectivity, common.Connectivity)
                    skip_value = connectivity.skip_value

                    # fold & convert expr into actual integers
                    range_exprs = new_ranges[old_dim].start, new_ranges[old_dim].stop
                    range_exprs = tuple(
                        collapse_tuple.CollapseTuple.apply(
                            expr,
                            within_stencil=False,
                            allow_undeclared_symbols=True,
                        )
                        for expr in range_exprs
                    )  # type: ignore[assignment]  # mypy not smart enough
                    assert all(isinstance(expr, itir.Literal) for expr in range_exprs)
                    start, stop = (int(literal.value) for literal in range_exprs)  # type: ignore[attr-defined]  # mypy does not understand assert above

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
                            UserWarning(
                                f"Translating '{self.as_expr()}' using '{shift[0].value}' has "
                                f"an out-of-bounds access."
                            ),
                            stacklevel=2,
                        )

                    new_start, new_stop = accessed.min(), accessed.max() + 1  # type: ignore[attr-defined]  # TODO(havogt): improve typing for NDArrayObject

                    fraction_accessed = np.unique(accessed).size / (new_stop - new_start)  # type: ignore[call-overload]  # TODO(havogt): improve typing for NDArrayObject

                    if (
                        fraction_accessed < NON_CONTIGUOUS_DOMAIN_WARNING_THRESHOLD
                        and (_NON_CONTIGUOUS_DOMAIN_WARNING_COUNTER := +1)
                        < NON_CONTIGUOUS_DOMAIN_MAX_WARNINGS
                    ):
                        warnings.warn(
                            UserWarning(
                                f"Translating '{self.as_expr()}' using '{shift[0].value}' requires "
                                f"computations on many additional points "
                                f"({round((1 - fraction_accessed) * 100)}%) in order to get a contiguous "
                                f"domain. Please consider reordering your mesh."
                            ),
                            stacklevel=2,
                        )

                    new_range = SymbolicRange(
                        im.literal(str(new_start), builtins.INTEGER_INDEX_BUILTIN),
                        im.literal(str(new_stop), builtins.INTEGER_INDEX_BUILTIN),
                    )

                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )
            else:
                raise AssertionError()
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
    start = functools.reduce(
        lambda current_expr, el_expr: start_reduce_op(current_expr, el_expr),
        [range_.start for range_ in ranges],
    )
    stop = functools.reduce(
        lambda current_expr, el_expr: stop_reduce_op(current_expr, el_expr),
        [range_.stop for range_ in ranges],
    )
    # constant fold expression to keep the tree small
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
