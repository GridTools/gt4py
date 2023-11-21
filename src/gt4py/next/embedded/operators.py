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

from typing import Callable, Sequence

from gt4py._core import definitions as core_defs
from gt4py.next import common, constructors, field_utils, utils
from gt4py.next.embedded import common as embedded_common


def scan(
    scan_op: Callable,
    forward: bool,
    init: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
    scan_axis: common.Dimension,
    scan_range: common.NamedRange,
    args: tuple,
    kwargs: dict,
):
    domain_intersection = embedded_common.intersect_domains(
        *[field_utils.get_domain(f) for f in [*args, *kwargs.values()] if _is_field_or_tuple(f)]
    )
    non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr[0] != scan_axis])

    res_domain = common.Domain(
        *[scan_range if nr[0] == scan_axis else nr for nr in domain_intersection]
    )
    if scan_axis not in res_domain.dims:
        # even if the scan dimension is not in the input, we can scan over it
        res_domain = common.Domain(*res_domain, (scan_range))

    res = _construct_scan_array(res_domain)(init)

    def combine_pos(hpos, vpos):
        hpos_iter = iter(hpos)
        return tuple(vpos if d == scan_axis else next(hpos_iter) for d in res_domain.dims)

    def scan_loop(hpos):
        acc = init
        for k in scan_range[1] if forward else reversed(scan_range[1]):
            pos = combine_pos(hpos, (scan_axis, k))
            new_args = [_tuple_at(pos)(arg) for arg in args]
            new_kwargs = {k: _tuple_at(pos)(v) for k, v in kwargs.items()}
            acc = scan_op(acc, *new_args, **new_kwargs)
            _tuple_assign(pos)(res, acc)

    for hpos in embedded_common.iterate_domain(non_scan_domain):
        scan_loop(hpos)
    if len(non_scan_domain) == 0:
        # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
        scan_loop(())

    return res


@utils.get_common_tuple_value
def _is_field_or_tuple(field: common.Field | tuple[common.Field, ...]) -> bool:
    return common.is_field(field)


def _construct_scan_array(domain: common.Domain):
    @utils.tree_map
    def impl(init: core_defs.Scalar) -> common.Field:
        return constructors.empty(domain, dtype=type(init))

    return impl


def _tuple_assign(
    pos: Sequence[common.NamedIndex],
) -> Callable[[common.MutableField | tuple[common.MutableField | tuple, ...]], None]:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar):
        target[pos] = source

    return impl  # type: ignore[return-value]


def _tuple_at(
    pos: Sequence[common.NamedIndex],
) -> Callable[
    [common.Field | core_defs.Scalar | tuple[common.Field | core_defs.Scalar | tuple, ...]],
    core_defs.Scalar,
]:
    @utils.tree_map
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos] if common.is_field(field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl  # type: ignore[return-value]
