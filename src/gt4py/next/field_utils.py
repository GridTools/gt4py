# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

from types import ModuleType
from typing import Callable

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common, utils
from gt4py.next.type_system import type_specifications as ts, type_translation


@utils.tree_map
def asnumpy(field: common.Field | np.ndarray) -> np.ndarray:
    return field.asnumpy() if isinstance(field, common.Field) else field


def field_from_typespec(
    type_: ts.TupleType | ts.ScalarType, domain: common.Domain, xp: ModuleType
) -> common.MutableField | tuple[common.MutableField | tuple, ...]:
    """
    Allocate a field or (arbitrarily nested) tuple(s) of fields.

    The tuple structure and dtype is taken from a type_specifications.DataType,
    which is either ScalarType or TupleType of ScalarType (possibly nested).

    >>> field_from_typespec(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT32), common.domain({common.Dimension("I"): 1}), np
    ... )  # doctest: +ELLIPSIS
    NumPyArrayField(... dtype=int32...)
    >>> field_from_typespec(
    ...     ts.TupleType(
    ...         types=[
    ...             ts.ScalarType(kind=ts.ScalarKind.INT32),
    ...             ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
    ...         ]
    ...     ),
    ...     common.domain({common.Dimension("I"): 1}),
    ...     np,
    ... )  # doctest: +ELLIPSIS
    (NumPyArrayField(... dtype=int32...), NumPyArrayField(... dtype=float32...))
    """

    @utils.tree_map(collection_type=ts.TupleType, result_collection_constructor=tuple)
    def impl(type_: ts.ScalarType) -> common.MutableField:
        res = common._field(
            xp.empty(domain.shape, dtype=xp.dtype(type_translation.as_dtype(type_).scalar_type)),
            domain=domain,
        )
        assert isinstance(res, common.MutableField)
        return res

    return impl(type_)



def get_array_ns(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> ModuleType:
    for arg in utils.flatten_nested_tuple(args):
        if hasattr(arg, "array_ns"):
            return arg.array_ns
    return np
