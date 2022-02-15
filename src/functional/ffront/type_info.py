from dataclasses import dataclass
from typing import Optional, Type, TypeGuard

from functional.ffront.common_types import (
    DeferredSymbolType,
    FieldType,
    ScalarKind,
    ScalarType,
    SymbolType,
)


def is_complete_symbol_type(fo_type: SymbolType) -> TypeGuard[SymbolType]:
    """Figure out if the foast type is completely deduced."""
    match fo_type:
        case None:
            return False
        case DeferredSymbolType():
            return False
        case SymbolType():
            return True
    return False


@dataclass
class TypeInfo:
    """
    Wrapper around foast types for type deduction and compatibility checks.

    Examples:
    ---------
    >>> type_a = ScalarType(kind=ScalarKind.FLOAT64)
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

    type: SymbolType  # noqa: A003

    @property
    def is_complete(self) -> bool:
        return is_complete_symbol_type(self.type)

    @property
    def is_any_type(self) -> bool:
        return (not self.is_complete) and ((self.type is None) or (self.constraint is None))

    @property
    def constraint(self) -> Optional[Type[SymbolType]]:
        """Find the constraint of a deferred type or the class of a complete type."""
        if self.is_complete:
            return self.type.__class__
        elif self.type:
            return self.type.constraint
        return None

    @property
    def is_field_type(self) -> bool:
        return issubclass(self.constraint, FieldType) if self.constraint else False

    @property
    def is_scalar(self) -> bool:
        return issubclass(self.constraint, ScalarType) if self.constraint else False

    @property
    def is_arithmetic_compatible(self) -> bool:
        match self.type:
            case FieldType(dtype=ScalarType(kind=dtype_kind)) | ScalarType(kind=dtype_kind):
                if dtype_kind is not ScalarKind.BOOL:
                    return True
        return False

    @property
    def is_logics_compatible(self) -> bool:
        match self.type:
            case FieldType(dtype=ScalarType(kind=dtype_kind)) | ScalarType(kind=dtype_kind):
                if dtype_kind is ScalarKind.BOOL:
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
