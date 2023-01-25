# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data Model field validators.

Check :mod:`eve.datamodels` and `attrs validation reference <https://www.attrs.org/en/stable/api.html#validators >`
for additional information.

Note that validators are implemented as callable classes just to be able to
customize their ``__repr__`` method.
"""

from __future__ import annotations

import typing
from typing import Any

import attrs
from attrs.validators import (
    and_,
    deep_iterable,
    deep_mapping,
    ge,
    gt,
    in_,
    instance_of,
    is_callable,
    le,
    lt,
    matches_re,
    max_len,
    optional,
)


if typing.TYPE_CHECKING:
    from .core import DataModelTP, FieldValidator


__all__ = [
    # reexported from attrs
    "and_",
    "deep_iterable",
    "deep_mapping",
    "ge",
    "gt",
    "in_",
    "instance_of",
    "is_callable",
    "le",
    "lt",
    "matches_re",
    "max_len",
    "optional",
    # custom
    "non_empty",
]


@attrs.define(repr=False, frozen=True, slots=True)
class _NonEmptyValidator:
    def __call__(self, inst: DataModelTP, attr: attrs.Attribute, value: Any) -> None:
        if not len(value):
            raise ValueError(f"Empty '{attr.name}' value")

    def __repr__(self) -> str:
        return "<non_empty validator>"


def non_empty() -> FieldValidator:
    """Create a validator for non-empty iterables."""
    return _NonEmptyValidator()
