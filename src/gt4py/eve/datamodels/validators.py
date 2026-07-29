# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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


__all__ = [  # noqa: RUF022 `__all__` is not sorted
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
