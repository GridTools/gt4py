import collections.abc
import typing

import gt4py.next
from gt4py.next.type_system_2 import inference as ti2
from gt4py.next.ffront.type_system_2 import types as ti2_f
import gt4py.next as gtx
from typing import Any


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
    return ti2_f.FieldType(element_type, set(dimensions))


def dimension_from_instance(_: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.Dimension):
        return ti2_f.DimensionType(instance)


def field_offset_from_instance(_: ti2.TypeInferrer, instance: Any):
    if isinstance(instance, gtx.FieldOffset):
        return ti2_f.FieldOffsetType(instance)


inferrer = ti2.TypeInferrer(
    [
        *ti2.inferrer.patterns,
        ti2.Pattern(field_from_annotation, None),
        ti2.Pattern(None, dimension_from_instance),
        ti2.Pattern(None, field_offset_from_instance),
    ]
)
