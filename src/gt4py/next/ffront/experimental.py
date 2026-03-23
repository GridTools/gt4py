# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from gt4py._core import definitions as core_defs
from gt4py.next import common, named_collections
from gt4py.next.ffront.fbuiltins import BuiltInFunction, FieldOffset, WhereBuiltinFunction


@BuiltInFunction
def as_offset(offset_: FieldOffset, field: common.Field, /) -> common.Connectivity:
    raise NotImplementedError()


@WhereBuiltinFunction
def concat_where(
    cond: common.Domain,
    true_field: common.Field | core_defs.ScalarT | Tuple | named_collections.CustomNamedCollection,
    false_field: common.Field | core_defs.ScalarT | Tuple | named_collections.CustomNamedCollection,
    /,
) -> common.Field | Tuple:
    """Assemble a field by selecting from ``true_field`` where ``cond`` applies and from ``false_field`` elsewhere.

    Unlike ``where`` (element-wise selection via a boolean mask field), ``concat_where``
    works on **domain regions**: the condition is a ``Domain`` (not a ``Field``), and the
    result is the concatenation of slices from the two fields along one dimension.
    Each field only needs to cover its own region — they may be non-overlapping.

    The condition must be a 1D ``Domain`` (e.g. ``I < 5``).

    Args:
        cond: 1D Domain specifying the "true" region.
        true_field: Field (or scalar) providing values inside the domain region.
        false_field: Field (or scalar) providing values outside the domain region.

    Returns:
        A new field whose domain is the concatenation of the contributed regions.

    Raises:
        NonContiguousDomain: If the resulting domain has interior gaps.
    """
    raise NotImplementedError()


EXPERIMENTAL_FUN_BUILTIN_NAMES = ["as_offset", "concat_where"]
