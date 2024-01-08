import dataclasses
import itertools
import typing
import copy

import gt4py.next.common as gtx_common
from gt4py.next.ffront import fbuiltins
from gt4py.next.type_system_2 import types as ts2, traits, utils as ts2_u
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
class ScanOperatorType(ts2.Type, traits.CallableTrait):
    dimension: gtx_common.Dimension
    carry: Optional[ts2.Type]
    parameters: list[ts2.FunctionParameter]
    result: Optional[ts2.Type]

    def __init__(
            self,
            dimension: gtx_common.Dimension,
            carry: Optional[ts2.Type],
            parameters: list[ts2.FunctionParameter],
            result: Optional[ts2.Type]
    ):
        self.dimension = dimension
        self.carry = carry
        self.parameters = parameters
        self.result = result

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str]:
        structures = [arg.ty for arg in args]
        flattened = [ts2_u.flatten_tuples(struct) for struct in structures]
        scalarized = [[get_element_type(ty) for ty in fl] for fl in flattened]
        restructured = [ts2_u.unflatten_tuples(scalar, struct) for scalar, struct in zip(scalarized, structures)]
        scalarized_args = [ts2.FunctionArgument(ty, arg.location) for ty, arg in zip(restructured, args)]
        return ts2.FunctionType(self.parameters, self.result).is_callable(scalarized_args)

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


@dataclasses.dataclass
class CastFunctionType(ts2.Type, traits.CallableTrait):
    result: ts2.Type

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        if len(args) != 1:
            return False, "expected exactly one argument"
        if not traits.is_convertible(args[0].ty, self.result):
            return False, f"could not convert '{args[0].ty}' to '{self.result}'"
        return True, self.result

    def __str__(self):
        return f"(Any) -> {self.result}"


@dataclasses.dataclass
class BuiltinFunctionType(ts2.Type, traits.CallableTrait):
    func: fbuiltins.BuiltInFunction

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        if self.func == fbuiltins.broadcast:
            return self.is_callable_broadcast(args)
        elif self.func == fbuiltins.astype:
            return self.is_callable_as_type(args)
        elif self.func == fbuiltins.neighbor_sum:
            return self.is_callable_reduction(args)
        elif self.func == fbuiltins.min_over:
            return self.is_callable_reduction(args)
        elif self.func == fbuiltins.max_over:
            return self.is_callable_reduction(args)
        return False, f"callable trait for '{self.func.function.__name__}' not implemented"

    def is_callable_broadcast(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        args = sorted(args, key=lambda v: v.location)
        if len(args) != 2:
            return False, "expected exactly two arguments"
        source, dims = args[0].ty, args[1].ty
        element_ty = get_element_type(source)
        source_dims = get_dimensions(source)
        if not isinstance(dims, ts2.TupleType) or not all(isinstance(dim, DimensionType) for dim in dims.elements):
            return False, "expected a tuple of dimensions for argument 2"
        target_dims = set(dim.dimension for dim in dims.elements)
        if not all(dim in target_dims for dim in source_dims):
            return False, f"broadcasting '{source}' to dimensions '{dims}' would remove dimensions"
        return True, FieldType(element_ty, target_dims)

    def is_callable_as_type(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        if len(args) != 2:
            return False, "expected an object and a type as arguments"
        value_arg, ty_args = args
        if not isinstance(ty_args.ty, CastFunctionType):
            return False, f"could not convert {ty_args.ty} to type literal"

        def can_cast_element(ty):
            element_ty = get_element_type(ty)
            success, result = ty_args.ty.is_callable([FunctionArgument(element_ty, 0)])
            if success:
                return success, FieldType(result, ty.dimensions) if isinstance(ty, FieldType) else result
            return success, result

        structure_ty = value_arg.ty
        source_tys = ts2_u.flatten_tuples(structure_ty)
        results = [can_cast_element(ty) for ty in source_tys]
        for success, message in results:
            if not success:
                return success, f"{message}; while converting {structure_ty} to {ty_args.ty.result}"
        result_tys = [r[1] for r in results]
        result_ty = ts2_u.unflatten_tuples(result_tys, structure_ty)
        return True, result_ty

    def is_callable_reduction(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        args_map = {arg.location: arg.ty for arg in args}
        if 0 not in args_map:
            return False, "expected a field as positional argument"
        if "axis" not in args_map:
            return False, "expected an 'axis' keyword argument"
        field_ty = args_map[0]
        axis = args_map["axis"]
        if not isinstance(field_ty, FieldType):
            return False, f"expected a field, got {field_ty}"
        if not isinstance(axis, DimensionType):
            return False, f"expected a dimension, got {axis}"
        if axis.dimension not in field_ty.dimensions:
            return False, f"field {field_ty} is missing reduction dimension {axis.dimension}"
        dimensions = {dim for dim in field_ty.dimensions if dim != axis.dimension}
        return True, FieldType(field_ty.element_type, dimensions)


