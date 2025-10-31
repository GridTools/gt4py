# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Generic, Optional, ParamSpec, Sequence, TypeVar

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common, errors, field_utils, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.field_utils import get_array_ns
from gt4py.next.type_system import type_specifications as ts, type_translation


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...], _P]):
    forward: bool
    init: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.get_closure_column_range()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = get_array_ns(*all_args)
        init_type = type_translation.from_value(self.init)
        assert isinstance(init_type, ts.TupleType | ts.ScalarType)
        res = field_utils.field_from_typespec(init_type, out_domain, xp)

        def scan_loop(hpos: Sequence[common.NamedIndex]) -> None:
            acc: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...] = self.init
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                pos = (*hpos, common.NamedIndex(scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                _tuple_assign_value(pos, res, acc)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res


def _get_out_domain(
    out: common.MutableField | tuple[common.MutableField | tuple, ...],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[f.domain for f in utils.flatten_nested_tuple((out,))]
    )


def field_operator_call(op: EmbeddedOperator[_R, _P], args: Any, kwargs: Any) -> Optional[_R]:
    if "out" in kwargs:
        # called from program or direct field_operator as program
        new_context_kwargs = {}
        if embedded_context.within_valid_context():
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

        out_domain = (
            common.normalize_domains(domain) if domain is not None else _get_out_domain(out)
        )

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.update(**new_context_kwargs):
            res = op(*args, **kwargs)
        _tuple_assign_field(
            out,
            res,  # type: ignore[arg-type] # maybe can't be inferred properly because decorator.py is not properly typed yet
            domain=out_domain,
        )
        return None
    else:
        # called from other field_operator or missing `out` argument
        if "offset_provider" in kwargs:
            # assuming we wanted to call the field_operator as program, otherwise `offset_provider` would not be there
            raise errors.MissingArgumentError(None, "out", True)
        return op(*args, **kwargs)


@utils.tree_map
def _get_vertical_range(domain: common.Domain) -> common.NamedRange | eve.NothingType:
    vertical_dim_filtered = [nr for nr in domain if nr.dim.kind == common.DimensionKind.VERTICAL]
    assert len(vertical_dim_filtered) <= 1
    return vertical_dim_filtered[0] if vertical_dim_filtered else eve.NOTHING


def _tuple_assign_field(
    target: tuple[common.MutableField | tuple, ...] | common.MutableField,
    source: tuple[common.Field | tuple, ...] | common.Field,
    domain: xtyping.MaybeNestedInTuple[common.Domain],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: common.Field, domain: common.Domain) -> None:
        if isinstance(source, common.Field):
            target[domain] = source[domain]
        else:
            assert core_defs.is_scalar_type(source)
            target[domain] = source

    if not isinstance(domain, tuple):
        domain = utils.tree_map(lambda _: domain)(target)
    impl(target, source, domain)


def _intersect_scan_args(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[arg.domain for arg in utils.flatten_nested_tuple(args) if isinstance(arg, common.Field)]
    )


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: common.MutableField | tuple[common.MutableField | tuple, ...],
    source: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar) -> None:
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: common.Field | core_defs.Scalar | tuple[common.Field | core_defs.Scalar | tuple, ...],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @utils.tree_map
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos].as_scalar() if isinstance(field, common.Field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
