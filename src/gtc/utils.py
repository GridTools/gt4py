# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import Any, Iterable, List, Tuple, Type


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
