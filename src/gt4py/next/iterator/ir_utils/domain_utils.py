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
from typing import Any, Literal, Mapping

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import trace_shifts


def _max_domain_sizes_by_location_type(offset_provider: Mapping[str, Any]) -> dict[str, int]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `origin_axis` and the maximum
    value inside the neighbor table to get the size of each `neighbor_axis`.
    """
    sizes = dict[str, int]()
    for provider in offset_provider.values():
        if isinstance(provider, gtx.NeighborTableOffsetProvider):
            assert provider.origin_axis.kind == gtx.DimensionKind.HORIZONTAL
            assert provider.neighbor_axis.kind == gtx.DimensionKind.HORIZONTAL
            sizes[provider.origin_axis.value] = max(
                sizes.get(provider.origin_axis.value, 0), provider.table.shape[0]
            )
            sizes[provider.neighbor_axis.value] = max(
                sizes.get(provider.neighbor_axis.value, 0),
                provider.table.max() + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )
    return sizes


@dataclasses.dataclass
class SymbolicRange:
    start: itir.Expr
    stop: itir.Expr

    def translate(self, distance: int) -> SymbolicRange:
        return SymbolicRange(im.plus(self.start, distance), im.plus(self.stop, distance))


@dataclasses.dataclass
class SymbolicDomain:
    grid_type: Literal["unstructured_domain", "cartesian_domain"]
    ranges: dict[
        common.Dimension, SymbolicRange
    ]  # TODO(havogt): remove `AxisLiteral` by `Dimension` everywhere

    @classmethod
    def from_expr(cls, node: itir.Node) -> SymbolicDomain:
        assert isinstance(node, itir.FunCall) and node.fun in [
            im.ref("unstructured_domain"),
            im.ref("cartesian_domain"),
        ]

        ranges: dict[common.Dimension, SymbolicRange] = {}
        for named_range in node.args:
            assert (
                isinstance(named_range, itir.FunCall)
                and isinstance(named_range.fun, itir.SymRef)
                and named_range.fun.id == "named_range"
            )
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, itir.AxisLiteral)

            ranges[common.Dimension(value=axis_literal.value, kind=axis_literal.kind)] = (
                SymbolicRange(lower_bound, upper_bound)
            )
        return cls(node.fun.id, ranges)  # type: ignore[attr-defined]  # ensure by assert above

    def as_expr(self) -> itir.FunCall:
        converted_ranges: dict[common.Dimension | str, tuple[itir.Expr, itir.Expr]] = {
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
        offset_provider: common.OffsetProvider,
    ) -> SymbolicDomain:
        dims = list(self.ranges.keys())
        new_ranges = {dim: self.ranges[dim] for dim in dims}
        if len(shift) == 0:
            return self
        if len(shift) == 2:
            off, val = shift
            assert isinstance(off, itir.OffsetLiteral) and isinstance(off.value, str)
            nbt_provider = offset_provider[off.value]
            if isinstance(nbt_provider, common.Dimension):
                if val is trace_shifts.Sentinel.VALUE:
                    raise NotImplementedError("Dynamic offsets not supported.")
                assert isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                current_dim = nbt_provider
                # cartesian offset
                new_ranges[current_dim] = SymbolicRange.translate(
                    self.ranges[current_dim], val.value
                )
            elif isinstance(nbt_provider, common.Connectivity):
                # unstructured shift
                assert (
                    isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                ) or val in [
                    trace_shifts.Sentinel.ALL_NEIGHBORS,
                    trace_shifts.Sentinel.VALUE,
                ]
                # note: ugly but cheap re-computation, but should disappear
                horizontal_sizes = _max_domain_sizes_by_location_type(offset_provider)

                old_dim = nbt_provider.origin_axis
                new_dim = nbt_provider.neighbor_axis

                assert new_dim not in new_ranges or old_dim == new_dim

                # TODO(tehrengruber): Do we need symbolic sizes, e.g., for ICON?
                new_range = SymbolicRange(
                    im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                    im.literal(str(horizontal_sizes[new_dim.value]), itir.INTEGER_INDEX_BUILTIN),
                )
                new_ranges = dict(
                    (dim, range_) if dim != old_dim else (new_dim, new_range)
                    for dim, range_ in new_ranges.items()
                )
            else:
                raise AssertionError()
            return SymbolicDomain(self.grid_type, new_ranges)
        elif len(shift) > 2:
            return self.translate(shift[0:2], offset_provider).translate(shift[2:], offset_provider)
        else:
            raise AssertionError("Number of shifts must be a multiple of 2.")


def domain_union(*domains: SymbolicDomain) -> SymbolicDomain:
    """Return the (set) union of a list of domains."""
    new_domain_ranges = {}
    assert all(domain.grid_type == domains[0].grid_type for domain in domains)
    assert all(domain.ranges.keys() == domains[0].ranges.keys() for domain in domains)
    for dim in domains[0].ranges.keys():
        start = functools.reduce(
            lambda current_expr, el_expr: im.call("minimum")(current_expr, el_expr),
            [domain.ranges[dim].start for domain in domains],
        )
        stop = functools.reduce(
            lambda current_expr, el_expr: im.call("maximum")(current_expr, el_expr),
            [domain.ranges[dim].stop for domain in domains],
        )
        new_domain_ranges[dim] = SymbolicRange(start, stop)

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)
