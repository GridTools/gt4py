# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from dataclasses import dataclass
from typing import Optional, Type, TypeGuard

from functional.ffront import common_types


def is_complete_symbol_type(fo_type: common_types.SymbolType) -> TypeGuard[common_types.SymbolType]:
    """Figure out if the foast type is completely deduced."""
    match fo_type:
        case None:
            return False
        case common_types.DeferredSymbolType():
            return False
        case common_types.SymbolType():
            return True
    return False


@dataclass
class TypeInfo:
    """
    Wrapper around foast types for type deduction and compatibility checks.

    Examples:
    ---------
    >>> type_a = common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64)
    >>> typeinfo_a = TypeInfo(type_a)
    >>> typeinfo_a.is_complete
    True
    >>> typeinfo_a.is_arithmetic_compatible
    True
    >>> typeinfo_a.is_logics_compatible
    False
    >>> typeinfo_b = TypeInfo(None)
    >>> typeinfo_b.is_any_type
    True
    >>> typeinfo_b.is_arithmetic_compatible
    False
    >>> typeinfo_b.can_be_refined_to(typeinfo_a)
    True

    """

    type: common_types.SymbolType  # noqa: A003

    @property
    def is_complete(self) -> bool:
        return is_complete_symbol_type(self.type)

    @property
    def is_any_type(self) -> bool:
        return (not self.is_complete) and ((self.type is None) or (self.constraint is None))

    @property
    def constraint(self) -> Optional[Type[common_types.SymbolType]]:
        """Find the constraint of a deferred type or the class of a complete type."""
        if self.is_complete:
            return self.type.__class__
        elif self.type:
            return self.type.constraint
        return None

    @property
    def is_field_type(self) -> bool:
        return issubclass(self.constraint, common_types.FieldType) if self.constraint else False

    @property
    def is_scalar(self) -> bool:
        return issubclass(self.constraint, common_types.ScalarType) if self.constraint else False

    @property
    def is_arithmetic_compatible(self) -> bool:
        match self.type:
            case common_types.FieldType(
                dtype=common_types.ScalarType(kind=dtype_kind)
            ) | common_types.ScalarType(kind=dtype_kind):
                if dtype_kind is not common_types.ScalarKind.BOOL:
                    return True
        return False

    @property
    def is_logics_compatible(self) -> bool:
        match self.type:
            case common_types.FieldType(
                dtype=common_types.ScalarType(kind=dtype_kind)
            ) | common_types.ScalarType(kind=dtype_kind):
                if dtype_kind is common_types.ScalarKind.BOOL:
                    return True
        return False

    def can_be_refined_to(self, other: "TypeInfo") -> bool:
        if self.is_any_type:
            return True
        if self.is_complete:
            return self.type == other.type
        if self.constraint:
            if other.is_complete:
                return isinstance(other.type, self.constraint)
            elif other.constraint:
                return self.constraint is other.constraint
        return False


def are_broadcast_compatible(left: TypeInfo, right: TypeInfo) -> bool:
    """
    Check if ``left`` and ``right`` types are compatible after optional broadcasting.

    A binary field operation between two arguments can proceed and the result is a field.
    on top of the dimensions, also the dtypes must match.

    Examples:
    ---------
    >>> int_scalar_t = TypeInfo(common_types.ScalarType(kind=common_types.ScalarKind.INT64))
    >>> are_broadcast_compatible(int_scalar_t, int_scalar_t)
    True
    >>> int_field_t = TypeInfo(common_types.FieldType(dtype=common_types.ScalarType(kind=common_types.ScalarKind.INT64),
    ...                         dims=...))
    >>> are_broadcast_compatible(int_field_t, int_scalar_t)
    True

    """
    if left.is_field_type and right.is_field_type:
        return left.type.dims == right.type.dims
    elif left.is_field_type and right.is_scalar:
        return left.type.dtype == right.type
    elif left.is_scalar and left.is_field_type:
        return left.type == right.type.dtype
    elif left.is_scalar and right.is_scalar:
        return left.type == right.type
    return False


def broadcast_typeinfos(left: TypeInfo, right: TypeInfo) -> TypeInfo:
    """
    Decide the result type of a binary operation between arguments of ``left`` and ``right`` type.

    Return None if the two types are not compatible even after broadcasting.

    Examples:
    ---------
    >>> int_scalar_t = TypeInfo(common_types.ScalarType(kind=common_types.ScalarKind.INT64))
    >>> int_field_t = TypeInfo(common_types.FieldType(dtype=common_types.ScalarType(kind=common_types.ScalarKind.INT64),
    ...                         dims=...))
    >>> assert broadcast_typeinfos(int_field_t, int_scalar_t).type == int_field_t.type

    """
    if not are_broadcast_compatible(left, right):
        return None
    if left.is_scalar and right.is_field_type:
        return right
    return left
