# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Iterator, Sequence

from gt4py.eve.extended_typing import Any, Optional, cast
from gt4py.next import common
from gt4py.next.embedded import exceptions as embedded_exceptions


def sub_domain(domain: common.Domain, index: common.AnyIndexSpec) -> common.Domain:
    index_sequence = common.as_any_index_sequence(index)

    if common.is_absolute_index_sequence(index_sequence):
        return _absolute_sub_domain(domain, index_sequence)

    if common.is_relative_index_sequence(index_sequence):
        return _relative_sub_domain(domain, index_sequence)

    raise IndexError(f"Unsupported index type: '{index}'.")


def _relative_sub_domain(
    domain: common.Domain, index: common.RelativeIndexSequence
) -> common.Domain:
    named_ranges: list[common.NamedRange] = []

    expanded = _expand_ellipsis(index, len(domain))
    if len(domain) < len(expanded):
        raise IndexError(
            f"Can not access dimension with index {index} of 'Field' with {len(domain)} dimensions."
        )
    expanded += (slice(None),) * (len(domain) - len(expanded))
    for (dim, rng), idx in zip(domain, expanded, strict=True):
        if isinstance(idx, slice):
            try:
                sliced = _slice_range(rng, idx)
                named_ranges.append(common.NamedRange(dim, sliced))
            except IndexError as ex:
                raise embedded_exceptions.IndexOutOfBounds(
                    domain=domain, indices=index, index=idx, dim=dim
                ) from ex
        else:
            # not in new domain
            assert common.is_int_index(idx)
            assert common.UnitRange.is_finite(rng)
            new_index = (rng.start if idx >= 0 else rng.stop) + idx
            if new_index < rng.start or new_index >= rng.stop:
                raise embedded_exceptions.IndexOutOfBounds(
                    domain=domain, indices=index, index=idx, dim=dim
                )

    return common.Domain(*named_ranges)


def _absolute_sub_domain(
    domain: common.Domain, index: common.AbsoluteIndexSequence
) -> common.Domain:
    named_ranges: list[common.NamedRange] = []
    for i, (dim, rng) in enumerate(domain):
        if (pos := _find_index_of_dim(dim, index)) is not None:
            named_idx = index[pos]
            _, idx = named_idx
            if isinstance(idx, common.UnitRange):
                if not idx <= rng:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )

                named_ranges.append(common.NamedRange(dim, idx))
            else:
                # not in new domain
                assert common.is_int_index(idx)
                if idx < rng.start or idx >= rng.stop:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )
        else:
            # dimension not mentioned in slice
            named_ranges.append(common.NamedRange(dim, domain.ranges[i]))

    return common.Domain(*named_ranges)


def domain_intersection(*domains: common.Domain) -> common.Domain:
    """
    Return the intersection of the given domains.

    Example:
        >>> I = common.Dimension("I")
        >>> domain_intersection(
        ...     common.domain({I: (0, 5)}), common.domain({I: (1, 3)})
        ... )  # doctest: +ELLIPSIS
        Domain(dims=(Dimension(value='I', ...), ranges=(UnitRange(1, 3),))
    """
    return functools.reduce(operator.and_, domains, common.Domain(dims=tuple(), ranges=tuple()))


def restrict_to_intersection(
    *domains: common.Domain,
    ignore_dims: Optional[common.Dimension | tuple[common.Dimension, ...]] = None,
) -> tuple[common.Domain, ...]:
    """
    Return the with each other intersected domains, ignoring 'ignore_dims' dimensions for the intersection.

    Example:
        >>> I = common.Dimension("I")
        >>> J = common.Dimension("J")
        >>> res = restrict_to_intersection(
        ...     common.domain({I: (0, 5), J: (1, 2)}),
        ...     common.domain({I: (1, 3), J: (0, 3)}),
        ...     ignore_dims=J,
        ... )
        >>> assert res == (
        ...     common.domain({I: (1, 3), J: (1, 2)}),
        ...     common.domain({I: (1, 3), J: (0, 3)}),
        ... )
    """
    ignore_dims_tuple = ignore_dims if isinstance(ignore_dims, tuple) else (ignore_dims,)
    intersection_without_ignore_dims = domain_intersection(
        *[
            common.Domain(*[nr for nr in domain if nr.dim not in ignore_dims_tuple])
            for domain in domains
        ]
    )
    return tuple(
        common.Domain(
            *[
                (nr if nr.dim in ignore_dims_tuple else intersection_without_ignore_dims[nr.dim])
                for nr in domain
            ]
        )
        for domain in domains
    )


def iterate_domain(domain: common.Domain) -> Iterator[tuple[common.NamedIndex]]:
    for idx in itertools.product(*(list(r) for r in domain.ranges)):
        yield tuple(common.NamedIndex(d, i) for d, i in zip(domain.dims, idx))  # type: ignore[misc] # trust me, `idx` is `tuple[int, ...]`


def _expand_ellipsis(
    indices: common.RelativeIndexSequence, target_size: int
) -> tuple[common.IntIndex | slice, ...]:
    if Ellipsis in indices:
        idx = indices.index(Ellipsis)
        indices = (
            indices[:idx] + (slice(None),) * (target_size - (len(indices) - 1)) + indices[idx + 1 :]
        )
    return cast(tuple[common.IntIndex | slice, ...], indices)  # mypy leave me alone and trust me!


def _slice_range(input_range: common.UnitRange, slice_obj: slice) -> common.UnitRange:
    if slice_obj == slice(None):
        return input_range

    start = (
        input_range.start if slice_obj.start is None or slice_obj.start >= 0 else input_range.stop
    ) + (slice_obj.start or 0)
    stop = (
        input_range.start if slice_obj.stop is None or slice_obj.stop >= 0 else input_range.stop
    ) + (slice_obj.stop or len(input_range))

    if start < input_range.start or stop > input_range.stop:
        raise IndexError("Slice out of range (no clipping following array API standard).")

    return common.UnitRange(start, stop)


def _find_index_of_dim(
    dim: common.Dimension,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex | Any],
) -> Optional[int]:
    for i, (d, _) in enumerate(domain_slice):
        if dim == d:
            return i
    return None


def canonicalize_any_index_sequence(index: common.AnyIndexSpec) -> common.AnyIndexSpec:
    # TODO: instead of canonicalizing to `NamedRange`, we should canonicalize to `NamedSlice`
    new_index: common.AnyIndexSpec = (index,) if isinstance(index, slice) else index
    if isinstance(new_index, tuple) and all(isinstance(i, slice) for i in new_index):
        new_index = tuple([_named_slice_to_named_range(i) for i in new_index])  # type: ignore[arg-type, assignment] # all i's are slices as per if statement
    return new_index


def _named_slice_to_named_range(idx: common.NamedSlice) -> common.NamedRange | common.NamedSlice:
    assert hasattr(idx, "start") and hasattr(idx, "stop")
    if common.is_named_slice(idx):
        start_dim, start_value = idx.start
        stop_dim, stop_value = idx.stop
        if start_dim != stop_dim:
            raise IndexError(
                f"Dimensions slicing mismatch between '{start_dim.value}' and '{stop_dim.value}'."
            )
        assert isinstance(start_value, int) and isinstance(stop_value, int)
        return common.NamedRange(start_dim, common.UnitRange(start_value, stop_value))
    if isinstance(idx.start, common.NamedIndex) and idx.stop is None:
        raise IndexError(f"Upper bound needs to be specified for {idx}.")
    if isinstance(idx.stop, common.NamedIndex) and idx.start is None:
        raise IndexError(f"Lower bound needs to be specified for {idx}.")
    return idx
