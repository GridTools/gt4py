# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Generic, Optional, ParamSpec, Sequence, TypeVar

import numpy as np

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import xtyping
from gt4py.next import common, errors, field_utils, named_collections, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.field_utils import get_array_ns
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_specifications as ts, type_translation


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[xtyping.MaybeNestedInTuple[core_defs.ScalarT], _P]):
    forward: bool
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT]
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

        xp = get_array_ns(*(arguments.extract(arg) for arg in all_args))
        init_type = type_translation.from_value(self.init)
        assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)
        res = field_utils.field_from_typespec(init_type, out_domain, xp)

        res_extracted = arguments.extract(res)
        assert xtyping.is_maybe_nested_in_tuple_of(res_extracted, common.MutableField)  # type: ignore[type-abstract]  # MutableField is abstract/generic
        views = _point_views(all_args, out_domain)
        targets = _point_targets(res_extracted, out_domain)
        if views is not None and targets is not None:
            self._scan_points(args, kwargs, views, targets, out_domain)
            return res

        def scan_loop(hpos: Sequence[common.NamedIndex]) -> None:
            acc: xtyping.MaybeNestedInTuple[core_defs.ScalarT] = self.init
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                pos = (*hpos, common.NamedIndex(scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                # convert custom NamedCollections to plain tuples for assignment
                acc_extracted = arguments.extract(acc)
                assert xtyping.is_maybe_nested_in_tuple_of(acc_extracted, core_defs.Scalar)  # type: ignore[arg-type]  # Scalar is a Union
                _tuple_assign_value(pos, res_extracted, acc_extracted)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res

    def _scan_points(
        self,
        args: Sequence[common.Field | core_defs.Scalar],
        kwargs: dict[str, common.Field | core_defs.Scalar],
        views: Sequence[Optional[np.ndarray]],
        targets: xtyping.MaybeNestedInTuple[np.ndarray],
        domain: common.Domain,
    ) -> None:
        scan_index = domain.dims.index(self.axis)
        shape = domain.shape
        n_levels = shape[scan_index]
        levels = range(n_levels) if self.forward else range(n_levels - 1, -1, -1)
        arg_views, kwarg_views = views[: len(args)], views[len(args) :]
        for column in np.ndindex(shape[:scan_index] + shape[scan_index + 1 :]):
            head, tail = column[:scan_index], column[scan_index:]
            acc: xtyping.MaybeNestedInTuple[core_defs.ScalarT] = self.init
            for k in levels:
                index = (*head, k, *tail)
                new_args = [a if v is None else v[index] for a, v in zip(args, arg_views)]
                new_kwargs = {
                    name: a if v is None else v[index]
                    for (name, a), v in zip(kwargs.items(), kwarg_views)
                }
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                acc_extracted = arguments.extract(acc)
                assert xtyping.is_maybe_nested_in_tuple_of(acc_extracted, core_defs.Scalar)  # type: ignore[arg-type]  # Scalar is a Union
                _assign_point(targets, acc_extracted, index)


def _point_views(
    values: Sequence[Any], domain: common.Domain
) -> Optional[list[Optional[np.ndarray]]]:
    """
    Each field of `values` as a NumPy array over `domain`, indexed by the position relative to it.

    Scalars map to `None`. Returns `None` if a value is neither a field nor a scalar, or a field
    cannot be read at every position of `domain`; reading point by point raises the error then.
    """
    if not common.Domain.is_finite(domain):
        return None
    views: list[Optional[np.ndarray]] = []
    for value in values:
        if not isinstance(value, common.Field):
            if not core_defs.is_scalar_type(value):
                return None
            views.append(None)
            continue
        field_dims = value.domain.dims
        if not all(d in domain.dims for d in field_dims):
            return None
        positions = [domain.dims.index(d) for d in field_dims]
        if positions != sorted(positions):
            return None
        array = value.asnumpy()
        key: list[slice | None] = []
        for dim, rng in zip(domain.dims, domain.ranges):
            if dim not in field_dims:
                key.append(None)
                continue
            i = field_dims.index(dim)
            field_range = value.domain.ranges[i]
            if not rng <= field_range:
                return None
            if array.shape[i] == 1 and not (
                common.UnitRange.is_finite(field_range) and len(field_range) == 1
            ):
                # a broadcast axis: its only element holds the value of the whole range
                key.append(slice(None))
            else:
                key.append(slice(rng.start - field_range.start, rng.stop - field_range.start))
        views.append(np.broadcast_to(array[tuple(key)], domain.shape))
    return views


def _point_targets(
    res: xtyping.MaybeNestedInTuple[common.MutableField], domain: common.Domain
) -> Optional[xtyping.MaybeNestedInTuple[np.ndarray]]:
    """The NumPy buffers of `res`, if all are NumPy arrays on exactly `domain`."""
    leaves = utils.flatten_nested_tuple((res,))
    if not all(isinstance(f.ndarray, np.ndarray) and f.domain == domain for f in leaves):
        return None
    return utils.tree_map(lambda f: f.ndarray)(res)


def _assign_point(
    target: xtyping.MaybeNestedInTuple[np.ndarray],
    value: xtyping.MaybeNestedInTuple[core_defs.Scalar],
    index: tuple[int, ...],
) -> None:
    if isinstance(target, tuple):
        assert isinstance(value, tuple) and len(value) == len(target)
        for t, v in zip(target, value):
            _assign_point(t, v, index)
    else:
        target[index] = value


def _get_out_domain(out: xtyping.MaybeNestedInTuple[common.MutableField]) -> common.Domain:
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

        # TODO(havogt): To do the assignment of the resulting fields we extract containers and act on plain tuples.
        # We currently apply the extract on both the rhs (`res`) computed by the operator and the lhs (`out`, provided by the user)
        # without checking if the types are consistent. However, these errors are caught in linting if enabled.
        container_extracted_out = arguments.extract(out)
        assert xtyping.is_maybe_nested_in_tuple_of(
            container_extracted_out,
            common.MutableField,  # type: ignore[type-abstract]  # MutableField is abstract/generic
        )
        out_domain = (
            utils.tree_map(common.domain)(domain)
            if domain is not None
            else _get_out_domain(container_extracted_out)
        )

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.update(**new_context_kwargs):
            res = op(*args, **kwargs)
        container_extracted_res = arguments.extract(res)  # type: ignore[arg-type] # TODO(havogt): see notes above
        _tuple_assign_field(container_extracted_out, container_extracted_res, domain=out_domain)  # type: ignore[arg-type]
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
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[common.Field],
    domain: xtyping.MaybeNestedInTuple[common.Domain],
) -> None:
    @named_collections.tree_map_named_collection
    def impl(target: common.MutableField, source: common.Field, domain: common.Domain) -> None:
        if isinstance(source, common.Field):
            target[domain] = source if source.domain == domain else source[domain]
        else:
            assert core_defs.is_scalar_type(source)
            target[domain] = source

    if not isinstance(domain, tuple):
        domain = named_collections.tree_map_named_collection(lambda _: domain)(target)  # type: ignore[assignment] # typing not precise enough
    impl(target, source, domain)


def _intersect_scan_args(
    *args: xtyping.MaybeNestedInTuple[core_defs.Scalar | common.Field],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[arg.domain for arg in utils.flatten_nested_tuple(args) if isinstance(arg, common.Field)]
    )


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[core_defs.Scalar],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar) -> None:
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: xtyping.MaybeNestedInTuple[common.Field | core_defs.Scalar],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @named_collections.tree_map_named_collection
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos].as_scalar() if isinstance(field, common.Field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
