# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.ffront.fbuiltins import BuiltInFunction, FieldOffset, WhereBuiltinFunction


@BuiltInFunction
def as_offset(offset_: FieldOffset, field: common.Field, /) -> common.Connectivity:
    raise NotImplementedError()


@WhereBuiltinFunction
def concat_where(
    mask: common.Field,
    true_field: common.Field | core_defs.ScalarT | Tuple,
    false_field: common.Field | core_defs.ScalarT | Tuple,
    /,
) -> common.Field | Tuple:
    """
    Concatenates two field fields based on a 1D mask.

    The resulting domain is the concatenation of the mask subdomains with the domains of the respective true or false fields.
    Empty domains at the beginning or end are ignored, but the interior must result in a consecutive domain.

    TODO(havogt): I can't get this doctest to run, even after copying the __doc__ in the decorator
    Example:
        >>> I = common.Dimension("I")
        >>> mask = common._field([True, False, True], domain={I: (0, 3)})
        >>> true_field = common._field([1, 2], domain={I: (0, 2)})
        >>> false_field = common._field([3, 4, 5], domain={I: (1, 4)})
        >>> assert concat_where(mask, true_field, false_field) == _field([1, 3], domain={I: (0, 2)})

        >>> mask = common._field([True, False, True], domain={I: (0, 3)})
        >>> true_field = common._field([1, 2, 3], domain={I: (0, 3)})
        >>> false_field = common._field(
        ...     [4], domain={I: (2, 3)}
        ... )  # error because of non-consecutive domain: missing I(1), but has I(0) and I(2) values
    """
    raise NotImplementedError()


EXPERIMENTAL_FUN_BUILTIN_NAMES = ["as_offset", "concat_where"]
