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
from typing import Any, Optional, TypeGuard, cast

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
    dimension_ranges: list[common.DimensionRange] = []

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
                dimension_ranges.append(common.DimensionRange(dim, sliced))
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

    return common.Domain(*dimension_ranges)


def _absolute_sub_domain(
    domain: common.Domain, index: common.CanonicalAbsoluteIndexSequence
) -> common.Domain:
    dimension_ranges: list[common.DimensionRange] = []
    for i, (dim, rng) in enumerate(domain):
        if (pos := _find_index_of_dim(dim, index)) is not None:
            named_idx = index[pos]
            idx = named_idx.value
            if isinstance(idx, common.UnitRange):
                clipped = _clip_open_bounds(idx, rng)
                if clipped is None:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )

                dimension_ranges.append(common.DimensionRange(dim, clipped))
            else:
                # not in new domain
                assert common.is_int_index(idx)
                if idx < rng.start or idx >= rng.stop:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )
        else:
            # dimension not mentioned in slice
            dimension_ranges.append(common.DimensionRange(dim, domain.ranges[i]))

    return common.Domain(*dimension_ranges)


def _clip_open_bounds(
    selection: common.UnitRange, bounds: common.UnitRange
) -> Optional[common.UnitRange]:
    """
    Replace the open (infinite) sides of `selection` by the corresponding sides of `bounds`.

    Indexing a field with an open bound selects up to the field's own bound on that side. Every
    infinite side is an open bound, also that of an infinite `Domain` or `DimensionRange` used
    as an index.

    Returns:
        The clipped range, or `None` if a closed bound of `selection` lies outside `bounds`.
    """
    start, stop = selection.start, selection.stop
    if start is common.Infinity.NEGATIVE:
        start = bounds.start
    elif not bounds.start <= start <= bounds.stop:
        return None
    if stop is common.Infinity.POSITIVE:
        stop = bounds.stop
    elif not bounds.start <= stop <= bounds.stop:
        return None
    return common.UnitRange(start, stop)


def domain_intersection(*domains: common.Domain) -> common.Domain:
    """
    Return the intersection of the given domains.

    Example:
        >>> class I(common.CartesianAxisIndex): ...
        >>> domain_intersection(
        ...     common.domain({I: (0, 5)}), common.domain({I: (1, 3)})
        ... )  # doctest: +ELLIPSIS
        Domain(dims=(gt4py.next.embedded.common:I[horizontal],), ranges=(UnitRange(1, 3),))
    """
    return functools.reduce(operator.and_, domains, common.Domain(dims=tuple(), ranges=tuple()))


def restrict_to_intersection(
    *domains: common.Domain,
    ignore_dims: Optional[common.Dimension | tuple[common.Dimension, ...]] = None,
) -> tuple[common.Domain, ...]:
    """
    Return the with each other intersected domains, ignoring 'ignore_dims' dimensions for the intersection.

    Example:
        >>> class I(common.CartesianAxisIndex): ...
        >>> class J(common.CartesianAxisIndex): ...
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


def iterate_domain(domain: common.Domain) -> Iterator[tuple[common.DimensionIndex]]:
    for idx in itertools.product(*(list(r) for r in domain.ranges)):
        yield tuple(d(i) for d, i in zip(domain.dims, idx))  # type: ignore[misc] # trust me, `idx` is `tuple[int, ...]`


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
    domain_slice: common.Domain | Sequence[common.DimensionRange | common.DimensionIndex | Any],
) -> Optional[int]:
    for i, elem in enumerate(domain_slice):
        if dim == elem.dim:
            return i
    return None


def canonicalize_any_index_sequence(index: common.AnyIndexSpec) -> common.AnyIndexSpec:
    # A `DimensionRange` is the canonical absolute form; a dimension slice is only input syntax.
    elements = index if isinstance(index, tuple) else (index,)
    if not any(_is_slice_with_dimension_bound(e) for e in elements):
        return index
    result: list[common.DimensionRange | common.DimensionIndex] = []
    for e in elements:
        if _is_slice_with_dimension_bound(e):
            result.append(_dimension_slice_to_range(e))
        elif isinstance(e, (common.DimensionRange, common.DimensionIndex)):
            result.append(e)
        else:
            raise IndexError(f"Cannot mix relative index '{e}' with absolute indices in '{index}'.")
    return tuple(result)


def _is_slice_with_dimension_bound(idx: object) -> TypeGuard[slice]:
    return isinstance(idx, slice) and (
        isinstance(idx.start, common.DimensionIndex) or isinstance(idx.stop, common.DimensionIndex)
    )


def _dimension_slice_to_range(idx: slice) -> common.DimensionRange:
    if idx.step not in (None, 1):
        raise IndexError(f"Invalid slice '{idx}': only a step of 1 is supported.")
    if not common.is_dimension_slice(idx):
        raise IndexError(
            f"Invalid slice '{idx}': both bounds must be indices on one dimension, or 'None' for"
            " an open bound."
        )
    start, stop = idx.start, idx.stop
    if start is not None and stop is not None and start.dim != stop.dim:
        raise IndexError(
            "Dimensions slicing mismatch between"
            f" '{start.dim.__qualname__}' and '{stop.dim.__qualname__}'."
        )
    bound = start if start is not None else stop
    assert bound is not None  # `is_dimension_slice` requires at least one bound
    return bound.dim.at(
        None if start is None else start.value, None if stop is None else stop.value
    )
