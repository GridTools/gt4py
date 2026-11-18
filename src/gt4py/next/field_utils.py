# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve.xtyping import NestedTuple
from gt4py.next import common, named_collections, utils
from gt4py.next.embedded import array_ops, nd_array_field
from gt4py.next.type_system import type_specifications as ts, type_translation


try:
    import cupy as cp
except ImportError:
    cp = None


@utils.tree_map
def asnumpy(field: common.Field | np.ndarray) -> np.ndarray:
    return field.asnumpy() if isinstance(field, common.Field) else field


def field_from_typespec(
    type_: ts.CollectionTypeSpec | ts.ScalarType,
    domain: common.Domain,
    *,
    like: core_defs.NDArrayObject | None = None,
) -> common.MutableField | tuple[common.MutableField | tuple, ...]:
    """
    Allocate a field or (arbitrarily nested) tuple(s) of fields.

    The buffers are allocated in the array namespace of `like`, on its device, or with
    NumPy if `like` is not given.

    The tuple structure and dtype is taken from a type_specifications.DataType,
    which is either ScalarType or a CollectionTypeSpec of ScalarType (possibly nested).

    >>> field_from_typespec(
    ...     ts.ScalarType(kind=ts.ScalarKind.INT32), common.domain({common.Dimension("I"): 1})
    ... )  # doctest: +ELLIPSIS
    NdArrayField(... dtype=int32...)
    >>> field_from_typespec(
    ...     ts.TupleType(
    ...         types=[
    ...             ts.ScalarType(kind=ts.ScalarKind.INT32),
    ...             ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
    ...         ]
    ...     ),
    ...     common.domain({common.Dimension("I"): 1}),
    ... )  # doctest: +ELLIPSIS
    (NdArrayField(... dtype=int32...), NdArrayField(... dtype=float32...))
    """
    ops = array_ops.of(like) if like is not None else array_ops.default()

    def _constructor(
        type_: ts.CollectionTypeSpec,
        elems: NestedTuple[common.MutableField],
    ) -> named_collections.NamedCollection:
        if isinstance(type_, ts.NamedCollectionType):
            return named_collections.make_named_collection_constructor_from_type_spec(type_)(elems)
        return tuple(elems)

    @utils.tree_map(
        collection_type=ts.COLLECTION_TYPE_SPECS,
        result_collection_constructor=_constructor,
    )
    def impl(type_: ts.ScalarType) -> common.MutableField:
        res = common._field(
            ops.empty(domain.shape, dtype=type_translation.as_dtype(type_).scalar_type, like=like),
            domain=domain,
        )
        assert isinstance(res, common.MutableField)
        return res

    return impl(type_)


def first_buffer(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> core_defs.NDArrayObject | None:
    """The buffer of the first buffer-backed field in the (nested tuples of) `args`, or `None`."""
    for arg in utils.flatten_nested_tuple(args):
        if isinstance(arg, nd_array_field.NdArrayField):
            return arg.ndarray
    return None


def verify_device_field_type(field: common.Field, device: core_defs.DeviceType) -> bool:
    """Check if `field` is suitable for `device`."""
    if not (array_ns := getattr(field, "array_ns", False)):
        return False  # not a NDArrayField
    if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]:
        assert core_defs.CUPY_DEVICE_TYPE is not None
        # TODO(havogt): generalize to other array libraries
        return device == core_defs.CUPY_DEVICE_TYPE and array_ns == cp
    else:
        return array_ns == np
