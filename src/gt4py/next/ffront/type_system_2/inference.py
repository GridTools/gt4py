import collections.abc
import typing

from gt4py.next.type_system_2 import inference as ti2, types as ts2
from gt4py.next.ffront.type_system_2 import types as ts2_f
import gt4py.next as gtx
from typing import Any
from gt4py.next.ffront import fbuiltins
from gt4py.next import common as gtx_common


def field_from_annotation(inferrer: ti2.TypeInferrer, annotation: Any):
    if typing.get_origin(annotation) != gtx.Field:
        return None
    args = typing.get_args(annotation)
    if len(args) != 2:
        raise ValueError("field type annotation: expected two arguments: dimensions, element type")
    dimensions = args[0]
    if (not isinstance(dimensions, collections.abc.Sequence) or
            not all(isinstance(dim, gtx.Dimension) for dim in dimensions)):
        raise ValueError("field type annotation: expected a list of dimensions")
    element_type = inferrer.from_annotation(args[1])
    if element_type is None:
        raise ValueError("field type annotation: expected a valid element_type")
    return ts2_f.FieldType(element_type, set(dimensions))


def field_from_instance(inferrer: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, gtx_common.Field):
        dimensions = instance.domain.dims
        element_type = inferrer.from_annotation(instance.dtype.dtype)
        if element_type is None:
            return None
        return ts2_f.FieldType(element_type, dimensions)
    return None


def field_operator_from_instance(_: ti2.TypeInferrer, instance: Any):
    from gt4py.next.ffront.decorator import FieldOperator
    if isinstance(instance, FieldOperator):
        return instance.foast_node.type_2
    return None


def dimension_from_instance(_: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.Dimension):
        return ts2_f.DimensionType(instance)


def field_offset_from_instance(_: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.FieldOffset):
        return ts2_f.FieldOffsetType(instance)


def cast_function_from_instance(inferrer: ti2.TypeInferrer, instance: Any):
    maybe_ty = inferrer.from_annotation(instance)
    if maybe_ty is not None:
        return ts2_f.CastFunctionType(maybe_ty)
    return None


def builtin_function_from_instance(_: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, fbuiltins.BuiltInFunction):
        return ts2_f.BuiltinFunctionType(instance)
    return None


inferrer = ti2.TypeInferrer(
    [
        *ti2.inferrer.patterns,
        ti2.Pattern(field_from_annotation, field_from_instance),
        ti2.Pattern(None, field_operator_from_instance),
        ti2.Pattern(None, dimension_from_instance),
        ti2.Pattern(None, field_offset_from_instance),
        ti2.Pattern(None, cast_function_from_instance),
        ti2.Pattern(None, builtin_function_from_instance),
    ]
)
