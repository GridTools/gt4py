# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import collections
from typing import Any, Iterable, List, Sequence, Tuple, Type


def listify(value):
    return value if isinstance(value, collections.abc.Sequence) else [value]


def flatten_list(
    nested_iterables: Iterable[Any],
    filter_none: bool = False,
    *,
    skip_types: Tuple[Type[Any], ...] = (str, bytes),
) -> List[Any]:
    return list(flatten_list_iter(nested_iterables, filter_none, skip_types=skip_types))


def flatten_list_iter(
    nested_iterables: Iterable[Any],
    filter_none: bool = False,
    *,
    skip_types: Tuple[Type[Any], ...] = (str, bytes),
) -> Any:
    for item in nested_iterables:
        if isinstance(item, list) and not isinstance(item, skip_types):
            yield from flatten_list(item, filter_none)
        else:
            if item is not None or not filter_none:
                yield item


def dimension_flags_to_names(mask: Tuple[bool, bool, bool]) -> str:
    labels = ["i", "j", "k"]
    selection = [i for i, flag in enumerate(mask) if flag]
    return "".join(labels[i] for i in selection)


def interpolate_mask(seq: Sequence[Any], mask: Sequence[bool], default) -> Tuple[Any, ...]:
    """
    Replace True values by those from the seq in the mask, else default.

    Example:
    >>> default = 0
    >>> a = (1, 2)
    >>> mask = (False, True, False, True)
    >>> interpolate_mask(a, mask, 0)
    (0, 1, 0, 2)
    """
    it = iter(seq)
    return tuple(next(it) if m else default for m in mask)


def filter_mask(seq: Sequence[Any], mask: Sequence[bool]) -> Tuple[Any, ...]:
    """
    Return a reduced-size tuple, with indices where mask[i]=False removed.

    Example:
    >>> a = (1, 2, 3)
    >>> mask = (False, True, False)
    >>> filter_mask(a, mask)
    (2,)
    """
    return tuple(s for m, s in zip(mask, seq) if m)
