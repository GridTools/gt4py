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
from typing import Any, Literal, Mapping, Optional

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding


def _max_domain_sizes_by_location_type(offset_provider: Mapping[str, Any]) -> dict[str, int]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `source_dim` and the maximum
    value inside the neighbor table to get the size of each `codomain`.
    """
    sizes = dict[str, int]()
    for provider in offset_provider.values():
        if common.is_neighbor_connectivity(provider):
            conn_type = provider.__gt_type__()
            sizes[conn_type.source_dim.value] = max(
                sizes.get(conn_type.source_dim.value, 0), provider.ndarray.shape[0]
            )
            sizes[conn_type.codomain.value] = max(
                sizes.get(conn_type.codomain.value, 0),
                provider.ndarray.max() + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
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
        #: A dictionary mapping axes names to their length. See
        #: func:`gt4py.next.iterator.transforms.infer_domain.infer_expr` for more details.
        symbolic_domain_sizes: Optional[dict[str, str]] = None,
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
            elif common.is_neighbor_connectivity(nbt_provider):
                # unstructured shift
                assert (
                    isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)
                ) or val in [
                    trace_shifts.Sentinel.ALL_NEIGHBORS,
                    trace_shifts.Sentinel.VALUE,
                ]
                horizontal_sizes: dict[str, itir.Expr]
                if symbolic_domain_sizes is not None:
                    horizontal_sizes = {k: im.ref(v) for k, v in symbolic_domain_sizes.items()}
                else:
                    # note: ugly but cheap re-computation, but should disappear
                    horizontal_sizes = {
                        k: im.literal(str(v), itir.INTEGER_INDEX_BUILTIN)
                        for k, v in _max_domain_sizes_by_location_type(offset_provider).items()
                    }

                old_dim = nbt_provider.__gt_type__().source_dim
                new_dim = nbt_provider.__gt_type__().codomain

                assert new_dim not in new_ranges or old_dim == new_dim

                new_range = SymbolicRange(
                    im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                    horizontal_sizes[new_dim.value],
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
        # constant fold expression to keep the tree small
        start, stop = ConstantFolding.apply(start), ConstantFolding.apply(stop)  # type: ignore[assignment]  # always an itir.FunCall
        new_domain_ranges[dim] = SymbolicRange(start, stop)

    return SymbolicDomain(domains[0].grid_type, new_domain_ranges)
