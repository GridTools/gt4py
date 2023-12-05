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

import dataclasses
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.next import common, constructors, field_utils, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[_R, _P]):
    forward: bool
    init: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...]
    axis: common.Dimension

    def __call__(self, *args: common.Field | core_defs.Scalar, **kwargs: common.Field | core_defs.Scalar) -> common.Field:  # type: ignore[override] # we cannot properly type annotate relative to self.fun
        scan_range = embedded_context.closure_column_range.get()
        assert self.axis == scan_range[0]

        return _scan(self.fun, self.forward, self.init, scan_range, args, kwargs)


def field_operator_call(op: EmbeddedOperator, args: Any, kwargs: Any):
    if "out" in kwargs:
        # called from program or direct field_operator as program
        offset_provider = kwargs.pop("offset_provider", None)

        new_context_kwargs = {}
        if embedded_context.within_context():
            # called from program
            assert offset_provider is None
        else:
            # field_operator as program
            new_context_kwargs["offset_provider"] = offset_provider

        out = kwargs.pop("out")
        domain = kwargs.pop("domain", None)
        out_domain = common.domain(domain) if domain is not None else field_utils.get_domain(out)

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.new_context(**new_context_kwargs) as ctx:
            res = ctx.run(op, *args, **kwargs)
            _tuple_assign_field(
                out,
                res,
                domain=out_domain,
            )
    else:
        # called from other field_operator
        return op(*args, **kwargs)


def _get_vertical_range(domain: common.Domain) -> common.NamedRange | eve.NothingType:
    vertical_dim_filtered = [nr for nr in domain if nr[0].kind == common.DimensionKind.VERTICAL]
    assert len(vertical_dim_filtered) <= 1
    return vertical_dim_filtered[0] if vertical_dim_filtered else eve.NOTHING


def _tuple_assign_field(
    target: tuple[common.MutableField | tuple, ...] | common.MutableField,
    source: tuple[common.Field | tuple, ...] | common.Field,
    domain: common.Domain,
):
    @utils.tree_map
    def impl(target: common.MutableField, source: common.Field):
        target[domain] = source[domain]

    impl(target, source)


def _scan(
    scan_op: Callable[_P, _R],
    forward: bool,
    init: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
    scan_range: common.NamedRange,
    args: tuple,
    kwargs: dict,
) -> Any:
    scan_axis = scan_range[0]
    domain_intersection = embedded_common.intersect_domains(
        *[_intersect_tuple_domain(f) for f in [*args, *kwargs.values()] if _is_field_or_tuple(f)]
    )
    non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr[0] != scan_axis])

    out_domain = common.Domain(
        *[scan_range if nr[0] == scan_axis else nr for nr in domain_intersection]
    )
    if scan_axis not in out_domain.dims:
        # even if the scan dimension is not in the input, we can scan over it
        out_domain = common.Domain(*out_domain, (scan_range))

    res = _construct_scan_array(out_domain)(init)

    def scan_loop(hpos):
        acc = init
        for k in scan_range[1] if forward else reversed(scan_range[1]):
            pos = (*hpos, (scan_axis, k))
            new_args = [_tuple_at(pos, arg) for arg in args]
            new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
            acc = scan_op(acc, *new_args, **new_kwargs)
            _tuple_assign_value(pos, res, acc)

    for hpos in embedded_common.iterate_domain(non_scan_domain):
        scan_loop(hpos)
    if len(non_scan_domain) == 0:
        # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
        scan_loop(())

    return res


@utils.tree_reduce(init=common.Domain())
def _intersect_tuple_domain(a: common.Domain, b: common.Field) -> common.Domain:
    return embedded_common.intersect_domains(a, b.domain)


@utils.get_common_tuple_value
def _is_field_or_tuple(field: common.Field) -> bool:
    return common.is_field(field)


def _construct_scan_array(domain: common.Domain):
    @utils.tree_map
    def impl(init: core_defs.Scalar) -> common.Field:
        return constructors.empty(domain, dtype=type(init))

    return impl


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: common.MutableField | tuple[common.MutableField | tuple, ...],
    source: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar):
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: common.Field | core_defs.Scalar | tuple[common.Field | core_defs.Scalar | tuple, ...],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @utils.tree_map
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos] if common.is_field(field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
