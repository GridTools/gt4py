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

from __future__ import annotations

from gt4py.eve.extended_typing import Any, Optional, Sequence, cast
from gt4py.next import common
from gt4py.next.embedded import exceptions as embedded_exceptions


def sub_domain(domain: common.Domain, index: common.AnyIndexSpec) -> common.Domain:
    index_sequence = common.as_any_index_sequence(index)

    if common.is_absolute_index_sequence(index_sequence):
        return _absolute_sub_domain(domain, index_sequence)

    if common.is_relative_index_sequence(index_sequence):
        return _relative_sub_domain(domain, index_sequence)

    raise IndexError(f"Unsupported index type: {index}")


def _relative_sub_domain(
    domain: common.Domain, index: common.RelativeIndexSequence
) -> common.Domain:
    named_ranges: list[common.NamedRange] = []

    expanded = _expand_ellipsis(index, len(domain))
    if len(domain) < len(expanded):
        raise IndexError(f"Trying to index a `Field` with {len(domain)} dimensions with {index}.")
    expanded += (slice(None),) * (len(domain) - len(expanded))
    for (dim, rng), idx in zip(domain, expanded, strict=True):
        if isinstance(idx, slice):
            try:
                sliced = _slice_range(rng, idx)
                named_ranges.append((dim, sliced))
            except IndexError:
                raise embedded_exceptions.IndexOutOfBounds(
                    domain=domain, indices=index, index=idx, dim=dim
                )
        else:
            # not in new domain
            assert common.is_int_index(idx)
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
            idx = named_idx[1]
            if isinstance(idx, common.UnitRange):
                if not idx <= rng:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )

                named_ranges.append((dim, idx))
            else:
                # not in new domain
                assert common.is_int_index(idx)
                if idx < rng.start or idx >= rng.stop:
                    raise embedded_exceptions.IndexOutOfBounds(
                        domain=domain, indices=index, index=named_idx, dim=dim
                    )
        else:
            # dimension not mentioned in slice
            named_ranges.append((dim, domain.ranges[i]))

    return common.Domain(*named_ranges)


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
        return common.UnitRange(input_range.start, input_range.stop)

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
