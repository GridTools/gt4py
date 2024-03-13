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
from types import ModuleType
from typing import Any, Callable, Generic, ParamSpec, Sequence, TypeVar

import numpy as np

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.next import common, errors, utils
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

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,  # type: ignore[override]
    ) -> common.Field:
        scan_range = embedded_context.closure_column_range.get()
        assert self.axis == scan_range[0]
        scan_axis = scan_range[0]
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr[0] != scan_axis])

        out_domain = common.Domain(*[
            scan_range if nr[0] == scan_axis else nr for nr in domain_intersection
        ])
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = _get_array_ns(*all_args)
        res = _construct_scan_array(out_domain, xp)(self.init)

        def scan_loop(hpos):
            acc = self.init
            for k in scan_range[1] if self.forward else reversed(scan_range[1]):
                pos = (*hpos, (scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)
                _tuple_assign_value(pos, res, acc)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res


def field_operator_call(op: EmbeddedOperator, args: Any, kwargs: Any):
    if "out" in kwargs:
        # called from program or direct field_operator as program
        new_context_kwargs = {}
        if embedded_context.within_context():
            # called from program
            assert "offset_provider" not in kwargs
        else:
            # field_operator as program
            if "offset_provider" not in kwargs:
                raise errors.MissingArgumentError(None, "offset_provider", True)
            offset_provider = kwargs.pop("offset_provider", None)

            new_context_kwargs["offset_provider"] = offset_provider

        out = kwargs.pop("out")

        domain = kwargs.pop("domain", None)

        flattened_out: tuple[common.Field, ...] = utils.flatten_nested_tuple((out,))
        assert all(f.domain == flattened_out[0].domain for f in flattened_out)

        out_domain = common.domain(domain) if domain is not None else flattened_out[0].domain

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.new_context(**new_context_kwargs) as ctx:
            res = ctx.run(op, *args, **kwargs)
            _tuple_assign_field(
                out,
                res,
                domain=out_domain,
            )
    else:
        # called from other field_operator or missing `out` argument
        if "offset_provider" in kwargs:
            # assuming we wanted to call the field_operator as program, otherwise `offset_provider` would not be there
            raise errors.MissingArgumentError(None, "out", True)
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
        if common.is_field(source):
            target[domain] = source[domain]
        else:
            assert core_defs.is_scalar_type(source)
            target[domain] = source

    impl(target, source)


def _intersect_scan_args(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> common.Domain:
    return embedded_common.intersect_domains(*[
        arg.domain for arg in utils.flatten_nested_tuple(args) if common.is_field(arg)
    ])


def _get_array_ns(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> ModuleType:
    for arg in utils.flatten_nested_tuple(args):
        if hasattr(arg, "array_ns"):
            return arg.array_ns
    return np


def _construct_scan_array(
    domain: common.Domain, xp: ModuleType
):  # TODO(havogt) introduce a NDArrayNamespace protocol
    @utils.tree_map
    def impl(init: core_defs.Scalar) -> common.Field:
        return common._field(xp.empty(domain.shape, dtype=type(init)), domain=domain)

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
        res = field[pos].as_scalar() if common.is_field(field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
