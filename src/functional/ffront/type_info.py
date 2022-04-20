from dataclasses import dataclass
from typing import Iterator, Optional, Type, TypeGuard, cast

from functional.common import GTTypeError
from functional.ffront.common_types import (
    DataType,
    DeferredSymbolType,
    FieldType,
    FunctionType,
    ScalarKind,
    ScalarType,
    SymbolType,
    TupleType,
    VoidType,
)


def is_complete_symbol_type(sym_type: Optional[SymbolType]) -> TypeGuard[SymbolType]:
    """Figure out if the foast type is completely deduced."""
    match sym_type:
        case None:
            return False
        case DeferredSymbolType():
            return False
        case SymbolType():
            return True
    return False


def is_complete_scalar_type(sym_type: Optional[SymbolType]) -> TypeGuard[ScalarType]:
    if isinstance(sym_type, ScalarType):
        return True
    return False


def function_signature_incompatibilities(
    func_type: FunctionType, args: list[SymbolType], kwargs: dict[str, SymbolType]
) -> Iterator[str]:
    """
    Return incompatibilities for a call to ``func_type`` with given arguments.

    Note that all types must be concrete/complete.
    """
    # check positional arguments
    if len(func_type.args) != len(args):
        yield f"Function takes {len(func_type.args)} arguments, but {len(args)} were given."
    for i, (a_arg, b_arg) in enumerate(zip(func_type.args, args)):
        if a_arg != b_arg:
            yield f"Expected {i}-th argument to be of type {a_arg}, but got {b_arg}."

    # check for missing or extra keyword arguments
    kw_a_m_b = set(func_type.kwargs.keys()) - set(kwargs.keys())
    if len(kw_a_m_b) > 0:
        yield f"Missing required keyword argument(s) `{'`, `'.join(kw_a_m_b)}`."
    kw_b_m_a = set(kwargs.keys()) - set(func_type.kwargs.keys())
    if len(kw_b_m_a) > 0:
        yield f"Got unexpected keyword argument(s) `{'`, `'.join(kw_b_m_a)}`."

    for kwarg in set(func_type.kwargs.keys()) & set(kwargs.keys()):
        if func_type.kwargs[kwarg] != kwargs[kwarg]:
            yield f"Expected keyword argument {kwarg} to be of type {func_type.kwargs[kwarg]}, but got {kwargs[kwarg]}."


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

    type: Optional[SymbolType]  # noqa: A003

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
            return cast(SymbolType, self.type).__class__
        elif self.type:
            return cast(DeferredSymbolType, self.type).constraint
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

    @property
    def is_callable(self) -> bool:
        return isinstance(self.type, FunctionType)

    @property
    def dims(self) -> Optional[list]:
        return getattr(self.type, "dims", None)

    @property
    def dtype(self) -> Optional[ScalarType]:
        if self.is_complete and self.is_scalar:
            return cast(ScalarType, self.type)
        if self.is_complete and self.is_field_type:
            return cast(FieldType, self.type).dtype
        return None

    @property
    def element_types(self) -> list[DataType]:
        if isinstance(self.type, TupleType):
            return self.type.types
        return []

    def is_callable_for_args(
        self,
        args: list[SymbolType],
        kwargs: dict[str, SymbolType],
        *,
        raise_exception: bool = False,
    ) -> bool:
        """
        Check if a function can be called for given arguments.

        If ``raise_exception`` is given a :class:`GTTypeError` is raised with a
        detailed description of why the function is not callable.

        Note that all types must be concrete/complete.

        Examples:
            >>> bool_type = ScalarType(kind=ScalarKind.BOOL)
            >>> func_type = FunctionType(
            ...     args=[bool_type],
            ...     kwargs={"foo": bool_type},
            ...     returns=VoidType()
            ... )
            >>> func_typeinfo = TypeInfo(func_type)
            >>> func_typeinfo.is_callable_for_args([bool_type], {"foo": bool_type})
            True
            >>> func_typeinfo.is_callable_for_args([], {})
            False
        """
        if not self.is_callable:
            if raise_exception:
                raise GTTypeError(f"Expected a function type, but got `{self.type}`.")
            return False

        errors = function_signature_incompatibilities(cast(FunctionType, self.type), args, kwargs)
        if raise_exception:
            error_list = list(errors)
            if len(error_list) > 0:
                raise GTTypeError(
                    f"Invalid call to function of type `{self.type}`:\n"
                    + ("\n".join([f"  - {error}" for error in error_list]))
                )
            return True

        return next(errors, None) is None

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
