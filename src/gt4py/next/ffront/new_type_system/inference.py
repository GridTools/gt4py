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

import collections.abc
import typing
from typing import Any

import gt4py.next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.ffront import fbuiltins
from gt4py.next.ffront.new_type_system import types as ts_f
from gt4py.next.new_type_system import inference as ti


def field_from_annotation(inferrer: ti.TypeInferrer, annotation: Any):
    if typing.get_origin(annotation) != gtx.Field:
        return None
    args = typing.get_args(annotation)
    if len(args) != 2:
        raise ValueError("field type annotation: expected two arguments: dimensions, element type")
    dimensions = args[0]
    if not isinstance(dimensions, collections.abc.Sequence) or not all(
        isinstance(dim, gtx.Dimension) for dim in dimensions
    ):
        raise ValueError("field type annotation: expected a list of dimensions")
    element_type = inferrer.from_annotation(args[1])
    if element_type is None:
        raise ValueError("field type annotation: expected a valid element_type")
    return ts_f.FieldType(element_type, set(dimensions))


def field_from_instance(inferrer: ti.TypeInferrer, instance: Any):
    if isinstance(instance, gtx_common.Field):  # type: ignore
        dimensions = instance.domain.dims
        element_type = inferrer.from_annotation(instance.dtype.dtype)
        if element_type is None:
            return None
        return ts_f.FieldType(element_type, set(dimensions))
    return None


def field_operator_from_instance(_: ti.TypeInferrer, instance: Any):
    from gt4py.next.ffront.decorator import FieldOperator

    if isinstance(instance, FieldOperator):
        return instance.foast_node.type_2
    return None


def dimension_from_instance(_: ti.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.Dimension):
        return ts_f.DimensionType(instance)


def field_offset_from_instance(_: ti.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.FieldOffset):
        return ts_f.FieldOffsetType(instance)


def cast_function_from_instance(inferrer: ti.TypeInferrer, instance: Any):
    maybe_ty = inferrer.from_annotation(instance)
    if maybe_ty is not None:
        return ts_f.CastFunctionType(maybe_ty)
    return None


def builtin_function_from_instance(_: ti.TypeInferrer, instance: Any):
    if isinstance(instance, fbuiltins.BuiltInFunction):
        return ts_f.BuiltinFunctionType(instance)
    return None


inferrer = ti.TypeInferrer(
    [
        *ti.inferrer.patterns,
        ti.Pattern(field_from_annotation, field_from_instance),
        ti.Pattern(None, field_operator_from_instance),
        ti.Pattern(None, dimension_from_instance),
        ti.Pattern(None, field_offset_from_instance),
        ti.Pattern(None, cast_function_from_instance),
        ti.Pattern(None, builtin_function_from_instance),
    ]
)
