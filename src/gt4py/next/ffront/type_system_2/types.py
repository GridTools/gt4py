import dataclasses
import typing
import copy

import gt4py.next.common as gtx_common
from gt4py.next.ffront import fbuiltins
from gt4py.next.type_system_2 import types as ts2, traits
from typing import Optional, Sequence, Any
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
    dimension: gtx_common.Dimension
    carry: Optional[ts2.Type]
    arguments: list[ts2.FunctionParameter]
    result: Optional[ts2.Type]

    def __init__(
            self,
            dimension: gtx_common.Dimension,
            carry: Optional[ts2.Type],
            arguments: list[ts2.FunctionParameter],
            result: Optional[ts2.Type]
    ):
        self.dimension = dimension
        self.carry = carry
        self.arguments = arguments
        self.result = result


@dataclasses.dataclass
class FieldType(
    ts2.Type,
    traits.FromTrait,
    traits.FromImplicitTrait,
    traits.CallableTrait,
    traits.BitwiseTrait,
    traits.ArithmeticTrait,
):
    element_type: ts2.Type
    dimensions: set[gtx_common.Dimension]

    def __init__(self, element_type: ts2.Type, dimensions: set[gtx_common.Dimension]):
        self.element_type = element_type
        self.dimensions = dimensions

    def __str__(self):
        dimensions = ", ".join(dim.value for dim in self.dimensions)
        element_type = str(self.element_type)
        return f"Field[[{dimensions}], {element_type}]"

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

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | ts2.Type]:
        if len(args) != 1 or not isinstance(args[0].ty, FieldOffsetType):
            return False, "expected a single argument of FieldOffset type"
        arg_t = typing.cast(FieldOffsetType, args[0].ty)
        result_dimensions = copy.copy(self.dimensions)
        result_dimensions.remove(arg_t.field_offset.source)
        for target_dim in arg_t.field_offset.target:
            result_dimensions.add(target_dim)
        return True, FieldType(self.element_type, result_dimensions)

    def supports_bitwise(self) -> bool:
        return isinstance(self.element_type, traits.BitwiseTrait) and self.element_type.supports_bitwise()

    def supports_arithmetic(self) -> bool:
        return isinstance(self.element_type, traits.ArithmeticTrait) and self.element_type.supports_arithmetic()

    def common_bitwise_type(self, other: ts2.Type) -> Optional[ts2.Type]:
        is_self_bitwise = self.supports_bitwise()
        is_other_bitwise = isinstance(other, traits.BitwiseTrait) and other.supports_bitwise()
        if not is_self_bitwise or not is_other_bitwise:
            return None
        assert isinstance(self.element_type, traits.BitwiseTrait)
        element_type = self.element_type.common_bitwise_type(get_element_type(other))
        dimensions = self.dimensions | get_dimensions(other)
        return FieldType(element_type, dimensions)

    def common_arithmetic_type(self, other: ts2.Type) -> Optional[ts2.Type]:
        is_self_arithmetic = self.supports_arithmetic()
        is_other_arithmetic = isinstance(other, traits.ArithmeticTrait) and other.supports_arithmetic()
        if not is_self_arithmetic or not is_other_arithmetic:
            return None
        assert isinstance(self.element_type, traits.ArithmeticTrait)
        element_type = self.element_type.common_arithmetic_type(get_element_type(other))
        dimensions = self.dimensions | get_dimensions(other)
        return FieldType(element_type, dimensions)


def get_element_type(ty: ts2.Type):
    if isinstance(ty, FieldType):
        return ty.element_type
    else:
        return ty


def get_dimensions(ty: ts2.Type):
    if isinstance(ty, FieldType):
        return ty.dimensions
    else:
        return set()


@dataclasses.dataclass
class DimensionType(ts2.Type):
    dimension: gtx_common.Dimension

    def __str__(self):
        return f"Dimension[{self.dimension.value}]"


@dataclasses.dataclass
class FieldOffsetType(ts2.Type):
    field_offset: fbuiltins.FieldOffset

    def __str__(self):
        return f"FieldOffset[{self.field_offset.value}]"
