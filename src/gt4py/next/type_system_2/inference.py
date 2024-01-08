from __future__ import annotations

import dataclasses
import typing

import numpy as np
from . import types
from typing import Any, Optional, Callable


@dataclasses.dataclass
class Pattern:
    annotation: Optional[Callable[[TypeInferrer, Any], Optional[types.Type]]]
    instance: Optional[Callable[[TypeInferrer, Any], Optional[types.Type]]]


@dataclasses.dataclass
class TypeInferrer:
    patterns: list[Pattern]

    def from_annotation(
            self,
            annotation: Any
    ) -> types.Type:
        for pattern in self.patterns:
            if pattern.annotation is not None:
                maybe_type = pattern.annotation(self, annotation)
                if maybe_type is not None:
                    return maybe_type

    def from_instance(
            self,
            instance: Any,
    ) -> types.Type:
        for pattern in self.patterns:
            if pattern.instance is not None:
                maybe_type = pattern.instance(self, instance)
                if maybe_type is not None:
                    return maybe_type


def primitive_from_annotation(_: TypeInferrer, annotation: Any) -> Optional[types.Type]:
    try:
        dtype = np.dtype(annotation)
        if not np.issctype(annotation):
            return None
        if dtype.kind == 'i':
            if dtype.itemsize not in [1, 2, 4, 8]:
                return None
            return types.IntegerType(8 * dtype.itemsize, True)
        if dtype.kind == 'u':
            if dtype.itemsize not in [1, 2, 4, 8]:
                return None
            return types.IntegerType(8 * dtype.itemsize, False)
        if dtype.kind == 'f':
            if dtype.itemsize not in [2, 4, 8]:
                return None
            return types.FloatType(8 * dtype.itemsize)
        if dtype.kind == 'b':
            return types.IntegerType(1, False)
    except:
        return None
    return None


def primitive_from_instance(inferrer: TypeInferrer, instance: Any) -> Optional[types.Type]:
    return primitive_from_annotation(inferrer, type(instance))


def tuple_from_annotation(inferrer: TypeInferrer, annotation: Any) -> Optional[types.TupleType]:
    if typing.get_origin(annotation) == tuple:
        elements = [inferrer.from_annotation(element) for element in typing.get_args(annotation)]
        if not all(elements):
            return None
        return types.TupleType(elements)


def tuple_from_instance(inferrer: TypeInferrer, instance: Any) -> Optional[types.TupleType]:
    if isinstance(instance, tuple):
        elements = [inferrer.from_instance(element) for element in instance]
        if not all(elements):
            return None
        return types.TupleType(elements)


inferrer = TypeInferrer(
    [
        Pattern(primitive_from_annotation, primitive_from_instance),
        Pattern(tuple_from_annotation, tuple_from_instance),
    ]
)