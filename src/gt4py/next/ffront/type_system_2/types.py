import dataclasses

from gt4py.next.type_system_2 import types as ts2, traits
from typing import Optional, Sequence, Any
from gt4py.next import common as func_common
from gt4py.next.type_system_2.traits import FunctionArgument


@dataclasses.dataclass
class FieldOperatorType(ts2.Type, traits.CallableTrait):
    parameters: list[ts2.FunctionParameter]
    result: Optional[ts2.Type]

    def __init__(self, parameters: list[ts2.FunctionParameter], result: Optional[ts2.Type]):
        self.parameters = parameters
        self.result = result

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str]:
        return ts2.FunctionType(self.parameters, self.result).is_callable(args)


@dataclasses.dataclass
class ScanOperatorType(ts2.Type):
    dimension: func_common.Dimension
    carry: Optional[ts2.Type]
    arguments: list[ts2.FunctionParameter]
    result: Optional[ts2.Type]

    def __init__(
            self,
            dimension: func_common.Dimension,
            carry: Optional[ts2.Type],
            arguments: list[ts2.FunctionParameter],
            result: Optional[ts2.Type]
    ):
        self.dimension = dimension
        self.carry = carry
        self.arguments = arguments
        self.result = result


@dataclasses.dataclass
class FieldType(ts2.Type, traits.FromTrait, traits.FromImplicitTrait):
    element_type: ts2.Type
    dimensions: set[func_common.Dimension]

    def __init__(self, element_type: ts2.Type, dimensions: set[func_common.Dimension]):
        self.element_type = element_type
        self.dimensions = dimensions

    def is_constructible_from(self, ty: ts2.Type) -> bool:
        if isinstance(ty, FieldType):
            element_convertible = traits.is_convertible(ty.element_type, self.element_type)
            dims_superset = all(dim in self.dimensions for dim in ty.dimensions)
            return element_convertible and dims_superset
        return traits.is_convertible(ty, self.element_type)

    def is_implicitly_constructible_from(self, ty: ts2.Type) -> bool:
        if isinstance(ty, FieldType):
            element_convertible = traits.is_implicitly_convertible(ty.element_type, self.element_type)
            dims_superset = all(dim in self.dimensions for dim in ty.dimensions)
            return element_convertible and dims_superset
        return traits.is_implicitly_convertible(ty, self.element_type)
