import enum
from dataclasses import dataclass
from typing import Iterator, Optional, Type, TypeGuard, cast

from functional.common import Dimension, GTTypeError
from functional.ffront import common_types as ct
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


def is_concrete(symbol_type: SymbolType) -> TypeGuard[SymbolType]:
    """Figure out if the foast type is completely deduced."""
    match symbol_type:
        case DeferredSymbolType():
            return False
        case SymbolType():
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


class TypeKind(enum.Enum):
    FIELD: 0
    SCALAR: 1
    UNKNOWN: 2


class UnknownDtype:
    ...


def type_kind(symbol_type: SymbolType) -> TypeKind:
    match symbol_type:
        case DeferredSymbolType(constraint=None):
            return TypeKind.UNKNOWN
        case TupleType(types as subtypes):
            if subtypes:
                return type_kind(subtypes[0])
        case FunctionType(returns as returntype):
            return type_kind(returntype)

    match type_class(symbol_type):
        case ct.FieldType:
            return TypeKind.FIELD
        case ct.ScalarType:
            return TypeKind.SCALAR

    return TypeKind.UNKNOWN


def type_class(symbol_type: SymbolType) -> Type[SymbolType]:
    match symbol_type:
        case DeferredSymbolType(constraint):
            if constraint is None:
                raise GTTypeError(f"No type information available!")
            return constraint
        case SymbolType() as concrete_type:
            return concrete_type.__class__
    raise GTTypeError(f"Invalid type for TypeInfo: requires {SymbolType}, got {type(self.type)}!")


def extract_dtype(symbol_type: SymbolType) -> ct.ScalarType:
    match symbol_type:
        case ct.FieldType(dtype):
            return dtype
        case ct.ScalarType() as dtype:
            return dtype
        case ct.FunctionType(returns=ct.FieldType(dtype)):
            return dtype
        case ct.FunctionType(returns=ct.ScalarType() as dtype):
            return dtype
    raise GTTypeError(f"Can not extract data type from {symbol_type}!")


def is_arithmetic(symbol_type: SymbolType) -> bool:
    if not isinstance(symbol_type, ScalarType):
        return False
    elif symbol_type.kind in [ct.ScalarKind.INT32, ct.ScalarKind.INT64, ct.ScalarKind.FLOAT32, ct.ScalarKind.FLOAT64]:
        return True
    return False
    

def is_locical(symbol_type: SymbolType) -> bool:
    if isinstance(symbol_type, ScalarType):
        return symbol_type.kind is ct.ScalarKind.BOOL
    return False


def extract_dims(symbol_type: SymbolType) -> list[Dimension]:
    match symbol_type:
        case ct.ScalarType():
            return []
        case ct.FieldType(dims):
            return dims
    raise GTTypeError(f"Can not extract dimensions from {symbol_type}!")


def can_concretize(symbol_type: SymbolType, to_type: SymbolType) -> SymbolType:
    if is_concrete(symbol_type):
        return symbol_type == to_type
    elif symbol_type.constraint == type_class(to_type) or symbol_type.constraint is None:
        return True
    return False


def can_broadcast()


class TypeInfo:
    """
    Wrapper around foast types for type deduction and compatibility checks.

    Examples:
    ---------
    >>> type_a = ScalarType(kind=ScalarKind.FLOAT64)
    >>> typeinfo_a = TypeInfo(type_a)
    >>> typeinfo_a.is_concrete
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

    type: ct.SymbolType  # noqa: A003

    @property
    def is_callable(self) -> bool:
        return isinstance(self.type, FunctionType)

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
        if self.is_concrete:
            return self.type == other.type
        if self.constraint:
            if other.is_concrete:
                return isinstance(other.type, self.constraint)
            elif other.constraint:
                return self.constraint is other.constraint
        return False
