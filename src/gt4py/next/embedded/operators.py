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

from typing import Any, Callable, Optional, Sequence

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.next import common, constructors, field_utils, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context


def field_operator_call(
    op: Callable, operator_attributes: Optional[dict[str, Any]], args: Any, kwargs: Any
):
    # embedded execution
    if embedded_context.within_context():
        # field_operator called from program or other field_operator in embedded execution
        if "out" in kwargs:
            # field_operator called from program in embedded execution

            offset_provider = kwargs.pop("offset_provider", None)
            # offset_provider should already be set
            assert (
                offset_provider is None or embedded_context.offset_provider.get() is offset_provider
            )
            out = kwargs.pop("out")
            domain = kwargs.pop("domain", None)
            out_domain = (
                common.domain(domain) if domain is not None else field_utils.get_domain(out)
            )

            vertical_range = _get_vertical_range(out_domain)

            with embedded_context.new_context(closure_column_range=vertical_range) as ctx:
                res = ctx.run(_run_operator, op, operator_attributes, args, kwargs)
                _tuple_assign_field(
                    out,
                    res,
                    domain=out_domain,
                )

        else:
            # field_operator called form field_operator in embedded execution
            return _run_operator(op, operator_attributes, args, kwargs)

    else:
        # field_operator called directly
        offset_provider = kwargs.pop("offset_provider", None)
        out = kwargs.pop("out")
        domain = kwargs.pop("domain", None)

        out_domain = common.domain(domain) if domain is not None else field_utils.get_domain(out)

        with embedded_context.new_context(
            offset_provider=offset_provider, closure_column_range=_get_vertical_range(out_domain)
        ) as ctx:
            res = ctx.run(_run_operator, op, operator_attributes, args, kwargs)
            _tuple_assign_field(
                out,
                res,
                domain=out_domain,
            )


def _run_operator(
    op: Callable, operator_attributes: Optional[dict[str, Any]], args: Any, kwargs: Any
):
    if operator_attributes is not None and any(
        has_scan_op_attribute := [
            attribute in operator_attributes for attribute in ["init", "axis", "forward"]
        ]
    ):
        assert all(has_scan_op_attribute)
        forward = operator_attributes["forward"]
        init = operator_attributes["init"]
        axis = operator_attributes["axis"]

        scan_range = embedded_context.closure_column_range.get()
        assert axis == scan_range[0]

        return scan(op, forward, init, scan_range, args, kwargs)
    else:
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


def scan(
    scan_op: Callable,
    forward: bool,
    init: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
    scan_range: common.NamedRange,
    args: tuple,
    kwargs: dict,
):
    scan_axis = scan_range[0]
    domain_intersection = embedded_common.intersect_domains(
        *[field_utils.get_domain(f) for f in [*args, *kwargs.values()] if _is_field_or_tuple(f)]
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
