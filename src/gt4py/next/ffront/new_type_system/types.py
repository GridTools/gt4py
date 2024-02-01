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

import copy
import dataclasses
import itertools
import typing
from typing import Optional, Sequence

import gt4py.next.common as gtx_common
from gt4py.next.ffront import fbuiltins
from gt4py.next.new_type_system import traits, types as ts, utils as ts_utils
from gt4py.next.new_type_system.traits import FunctionArgument


@dataclasses.dataclass
class FieldOperatorType(ts.Type, traits.CallableTrait):
    parameters: list[ts.FunctionParameter]
    result: Optional[ts.Type]

    def __init__(self, parameters: list[ts.FunctionParameter], result: Optional[ts.Type]):
        self.parameters = parameters
        self.result = result

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        return ts.FunctionType(self.parameters, self.result).is_callable(args)


@dataclasses.dataclass
class ScanOperatorType(ts.Type, traits.CallableTrait):
    dimension: gtx_common.Dimension
    carry: Optional[ts.Type]
    parameters: list[ts.FunctionParameter]
    result: Optional[ts.Type]

    def __init__(
        self,
        dimension: gtx_common.Dimension,
        carry: Optional[ts.Type],
        parameters: list[ts.FunctionParameter],
        result: Optional[ts.Type],
    ):
        self.dimension = dimension
        self.carry = carry
        self.parameters = parameters
        self.result = result

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        structures = [arg.ty for arg in args]
        flats = [ts_utils.flatten_tuples(struct) for struct in structures]
        flat_scalars = [[get_element_type(ty) for ty in fl] for fl in flats]
        scalars = [
            ts_utils.unflatten_tuples(scalar, struct)
            for scalar, struct in zip(flat_scalars, structures)
        ]
        scalar_args = [ts.FunctionArgument(ty, arg.location) for ty, arg in zip(scalars, args)]
        valid = ts.FunctionType(self.parameters, self.result).is_callable(scalar_args)
        if not valid:
            return valid
        dimensions = list(itertools.chain(*[[get_dimensions(ty) for ty in fl] for fl in flats]))
        merged = set(itertools.chain(*dimensions))
        if merged:
            return traits.CallValidity(FieldType(valid.result, merged))
        return valid


@dataclasses.dataclass
class FieldType(
    ts.Type,
    traits.FromTrait,
    traits.FromImplicitTrait,
    traits.CallableTrait,
    traits.BitwiseTrait,
    traits.ArithmeticTrait,
):
    element_type: ts.Type
    dimensions: set[gtx_common.Dimension]

    def __init__(self, element_type: ts.Type, dimensions: set[gtx_common.Dimension]):
        self.element_type = element_type
        self.dimensions = dimensions

    def __str__(self):
        dimensions = ", ".join(sorted(dim.value for dim in self.dimensions))
        element_type = str(self.element_type)
        return f"Field[[{dimensions}], {element_type}]"

    def is_constructible_from(self, ty: ts.Type) -> bool:
        if isinstance(ty, FieldType):
            element_convertible = traits.is_convertible(ty.element_type, self.element_type)
            dims_superset = all(dim in self.dimensions for dim in ty.dimensions)
            return element_convertible and dims_superset
        return traits.is_convertible(ty, self.element_type)

    def is_implicitly_constructible_from(self, ty: ts.Type) -> bool:
        if isinstance(ty, FieldType):
            element_convertible = traits.is_implicitly_convertible(
                ty.element_type, self.element_type
            )
            dims_superset = all(dim in self.dimensions for dim in ty.dimensions)
            return element_convertible and dims_superset
        return traits.is_implicitly_convertible(ty, self.element_type)

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        if len(args) != 1 or not isinstance(args[0].ty, FieldOffsetType):
            return traits.CallValidity(["expected a single argument of FieldOffset type"])
        arg_t = args[0].ty
        if arg_t.field_offset.source not in self.dimensions:
            return traits.CallValidity(["source dimension not in field"])
        result_dimensions = copy.copy(self.dimensions)
        result_dimensions.remove(arg_t.field_offset.source)
        for target_dim in arg_t.field_offset.target:
            result_dimensions.add(target_dim)
        return traits.CallValidity(FieldType(self.element_type, result_dimensions))

    def supports_bitwise(self) -> bool:
        return (
            isinstance(self.element_type, traits.BitwiseTrait)
            and self.element_type.supports_bitwise()
        )

    def supports_arithmetic(self) -> bool:
        return (
            isinstance(self.element_type, traits.ArithmeticTrait)
            and self.element_type.supports_arithmetic()
        )

    def common_bitwise_type(self, other: ts.Type) -> Optional[ts.Type]:
        is_self_bitwise = self.supports_bitwise()
        is_other_bitwise = isinstance(other, traits.BitwiseTrait) and other.supports_bitwise()
        if not is_self_bitwise or not is_other_bitwise:
            return None
        assert isinstance(self.element_type, traits.BitwiseTrait)
        element_type = self.element_type.common_bitwise_type(get_element_type(other))
        if element_type is None:
            return None
        dimensions = self.dimensions | get_dimensions(other)
        return FieldType(element_type, dimensions)

    def common_arithmetic_type(self, other: ts.Type) -> Optional[ts.Type]:
        is_self_arithmetic = self.supports_arithmetic()
        is_other_arithmetic = (
            isinstance(other, traits.ArithmeticTrait) and other.supports_arithmetic()
        )
        if not is_self_arithmetic or not is_other_arithmetic:
            return None
        assert isinstance(self.element_type, traits.ArithmeticTrait)
        element_type = self.element_type.common_arithmetic_type(get_element_type(other))
        if element_type is None:
            return None
        dimensions = self.dimensions | get_dimensions(other)
        return FieldType(element_type, dimensions)


def get_element_type(ty: ts.Type):
    if isinstance(ty, FieldType):
        return ty.element_type
    else:
        return ty


def get_dimensions(ty: ts.Type):
    if isinstance(ty, FieldType):
        return ty.dimensions
    else:
        return set()


@dataclasses.dataclass
class DimensionType(ts.Type):
    dimension: gtx_common.Dimension

    def __str__(self):
        return f"Dimension[{self.dimension.value}]"


@dataclasses.dataclass
class FieldOffsetType(ts.Type):
    field_offset: fbuiltins.FieldOffset

    def __str__(self):
        return f"FieldOffset[{self.field_offset.value}]"


@dataclasses.dataclass
class CastFunctionType(ts.Type, traits.CallableTrait):
    result: ts.Type

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        if len(args) != 1:
            return traits.CallValidity(["expected exactly one argument"])
        if not traits.is_convertible(args[0].ty, self.result):
            return traits.CallValidity(["could not convert '{args[0].ty}' to '{self.result}'"])
        return traits.CallValidity(self.result)

    def __str__(self):
        return f"(To[{self.result}]) -> {self.result}"


@dataclasses.dataclass
class BuiltinFunctionType(ts.Type, traits.CallableTrait):
    func: fbuiltins.BuiltInFunction

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
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
        elif self.func.name.split(".")[-1] == "minimum":
            return self.is_callable_arithmetic_binop(args)
        elif self.func.name.split(".")[-1] == "maximum":
            return self.is_callable_arithmetic_binop(args)
        elif self.func == fbuiltins.where:
            return self.is_callable_where(args)
        return traits.CallValidity(
            [f"callable trait for '{self.func.function.__name__}' not implemented"]
        )

    def is_callable_broadcast(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        args = sorted(args, key=lambda v: v.location)
        if len(args) != 2:
            return traits.CallValidity(["expected exactly two arguments"])
        source, dims = args[0].ty, args[1].ty
        element_ty = get_element_type(source)
        source_dims = get_dimensions(source)
        if not isinstance(dims, ts.TupleType) or not all(
            isinstance(dim, DimensionType) for dim in dims.elements
        ):
            return traits.CallValidity(["expected a tuple of dimensions for argument 2"])
        target_dims = set(typing.cast(DimensionType, dim).dimension for dim in dims.elements)
        if not all(dim in target_dims for dim in source_dims):
            return traits.CallValidity(
                [f"broadcasting '{source}' to dimensions '{dims}' would remove dimensions"]
            )
        return traits.CallValidity(FieldType(element_ty, target_dims))

    def is_callable_as_type(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        if len(args) != 2:
            return traits.CallValidity(["expected an object and a type as arguments"])
        value_arg, ty_args = args
        if not isinstance(ty_args.ty, CastFunctionType):
            return traits.CallValidity([f"could not convert {ty_args.ty} to type literal"])

        def can_cast_element(ty):
            element_ty = get_element_type(ty)
            valid: traits.CallValidity = ty_args.ty.is_callable([FunctionArgument(element_ty, 0)])
            if valid:
                return traits.CallValidity(
                    FieldType(valid.result, ty.dimensions)
                    if isinstance(ty, FieldType)
                    else valid.result
                )
            return valid

        structure_ty = value_arg.ty
        source_tys = ts_utils.flatten_tuples(structure_ty)
        results = [can_cast_element(ty) for ty in source_tys]
        for valid in results:
            if not valid:
                return traits.CallValidity(
                    [
                        f"{', '.join(valid.errors)}; while converting {structure_ty} to {ty_args.ty.result}"
                    ]
                )
        result_tys = [r.result for r in results]
        result_ty = ts_utils.unflatten_tuples(result_tys, structure_ty)
        return traits.CallValidity(result_ty)

    def is_callable_reduction(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        args_map = {arg.location: arg.ty for arg in args}
        if 0 not in args_map:
            return traits.CallValidity(["expected a field as positional argument"])
        if "axis" not in args_map:
            return traits.CallValidity(["expected an 'axis' keyword argument"])
        field_ty = args_map[0]
        axis = args_map["axis"]
        if not isinstance(field_ty, FieldType):
            return traits.CallValidity([f"expected a field, got {field_ty}"])
        if not isinstance(axis, DimensionType):
            return traits.CallValidity([f"expected a dimension, got {axis}"])
        if axis.dimension.kind != gtx_common.DimensionKind.LOCAL:
            return traits.CallValidity(
                [f"reduction dimension {axis.dimension} must be a local dimension"]
            )
        if axis.dimension not in field_ty.dimensions:
            return traits.CallValidity(
                [f"field {field_ty} is missing reduction dimension {axis.dimension}"]
            )
        dimensions = {dim for dim in field_ty.dimensions if dim != axis.dimension}
        return traits.CallValidity(FieldType(field_ty.element_type, dimensions))

    def is_callable_arithmetic_binop(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        if len(args) != 2:
            return traits.CallValidity([f"expected 2 arguments, got {len(args)}"])
        lhs, rhs = args
        if not isinstance(lhs.ty, traits.ArithmeticTrait) or not lhs.ty.supports_arithmetic():
            return traits.CallValidity([f"expected arithmetic type for LHS, got '{lhs.ty}'"])
        if not isinstance(rhs.ty, traits.ArithmeticTrait) or not rhs.ty.supports_arithmetic():
            return traits.CallValidity([f"expected arithmetic type for RHS, got '{rhs.ty}'"])
        common_ty = rhs.ty.common_arithmetic_type(lhs.ty)
        if not common_ty:
            return traits.CallValidity(
                [f"could not promote '{lhs.ty}' and '{lhs.ty}' to a common arithmetic type"]
            )
        return traits.CallValidity(common_ty)

    def is_callable_where(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        if len(args) != 3:
            return traits.CallValidity([f"expected 3 arguments, got {len(args)}"])
        cond, lhs, rhs = args
        cond_elem = get_element_type(cond.ty)
        lhs_elem = get_element_type(lhs.ty)
        rhs_elem = get_element_type(rhs.ty)

        combined_dims = get_dimensions(cond.ty) | get_dimensions(lhs.ty) | get_dimensions(rhs.ty)
        is_field = (
            isinstance(cond.ty, FieldType)
            or isinstance(lhs.ty, FieldType)
            or isinstance(rhs.ty, FieldType)
        )

        if not traits.is_implicitly_convertible(cond_elem, ts.BoolType()):
            return traits.CallValidity(
                [
                    (
                        f"could not implicitly convert from '{cond_elem}' to '{ts.BoolType()}'"
                        f" (condition was '{cond.ty}')"
                    )
                ]
            )
        common_ty = traits.common_type(lhs_elem, rhs_elem)
        if not common_ty:
            return traits.CallValidity(
                [
                    (
                        f"could not convert '{lhs_elem}' and '{lhs_elem}' to a common type"
                        f" (LHS was {lhs.ty} and RHS was {rhs.ty})"
                    )
                ]
            )
        if not is_field:
            return traits.CallValidity(common_ty)
        else:
            return traits.CallValidity(FieldType(common_ty, combined_dims))
