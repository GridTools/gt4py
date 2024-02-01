from . import traits
import dataclasses
from typing import Optional, Any, Sequence

from .traits import FunctionArgument


class Type:
    ...


@dataclasses.dataclass
class IntegerType(
    Type,
    traits.SignednessTrait,
    traits.FromTrait,
    traits.FromImplicitTrait,
    traits.ArithmeticTrait,
    traits.BitwiseTrait
):
    """
    Integer type.

    The types can have all standard power of two widths up to 64 bit, and may
    be signed or unsigned. Booleans are represented by a width of 1. Booleans
    cannot be signed.
    """

    width: int
    """The number of bits of the integer type."""
    signed: bool
    """Whether the type is signed or unsigned."""

    def __init__(self, width: int, signed: bool):
        assert width in [1, 8, 16, 32, 64]
        if width == 1:
            assert signed == False, "booleans must be unsigned"
        self.width = width
        self.signed = signed

    def __str__(self):
        signed = "" if self.signed else "u"
        return f"{signed}int{self.width}" if self.width > 1 else "bool"

    def __eq__(self, other):
        if isinstance(other, IntegerType):
            return self.width == other.width and self.signed == other.signed
        return False

    def is_signed(self) -> bool:
        return self.signed

    def is_constructible_from(self, ty: Type) -> bool:
        return isinstance(ty, IntegerType) or isinstance(ty, FloatType)

    def is_implicitly_constructible_from(self, ty: Any) -> bool:
        if isinstance(ty, IntegerType):
            if self.signed:
                return (ty.signed and ty.width <= self.width) or (not ty.signed and ty.width < self.width)
            else:
                return not ty.signed and ty.width <= self.width
        return False

    def common_arithmetic_type(self, other: Type) -> Optional[Type]:
        return self.common_bitwise_type(other)

    def common_bitwise_type(self, other: Type) -> Optional[Type]:
        if isinstance(other, IntegerType):
            if not self.signed and other.signed:
                return None
            elif self.signed and not other.signed:
                required_width = max(self.width, 2 * other.width)
                return IntegerType(required_width, True) if required_width <= 64 else None
            else:
                required_width = max(self.width, other.width)
                return IntegerType(required_width, self.signed)
        return None


@dataclasses.dataclass
class FloatType(
    Type,
    traits.SignednessTrait,
    traits.FromTrait,
    traits.FromImplicitTrait,
    traits.ArithmeticTrait
):
    """
    Floating point type.

    The width can be any power of two up to 64 bits.
    """

    width: int
    """The number of bits of the floating point type."""

    def __init__(self, width: int):
        assert (width in [16, 32, 64])
        self.width = width

    def __str__(self):
        return f"float{self.width}"

    def __eq__(self, other):
        if isinstance(other, FloatType):
            return self.width == other.width
        return False

    def is_signed(self) -> bool:
        return True

    def is_constructible_from(self, ty: Type) -> bool:
        return isinstance(ty, IntegerType) or isinstance(ty, FloatType)

    def is_implicitly_constructible_from(self, ty: Any) -> bool:
        if isinstance(ty, IntegerType):
            return ty.width < self.width
        elif isinstance(ty, FloatType):
            return ty.width <= self.width
        return False

    def common_arithmetic_type(self, other: Type) -> Optional[Type]:
        if isinstance(other, IntegerType):
            required_width = max(self.width, 2 * other.width)
            return FloatType(required_width) if required_width <= 64 else None
        elif isinstance(other, FloatType):
            required_width = max(self.width, other.width)
            return FloatType(required_width)
        return None


@dataclasses.dataclass
class TupleType(Type, traits.FromTrait, traits.FromImplicitTrait):
    """A usual tuple type that consists of an ordered list of elements."""

    elements: list[Type]
    """
    The types of the elements of the tuple. Ordering of the elements matters.
    """

    def __init__(self, elements: list[Type]):
        self.elements = elements

    def __str__(self):
        elements = ", ".join(str(element) for element in self.elements)
        return f"tuple[{elements}]"

    def is_constructible_from(self, ty: Type) -> bool:
        if not isinstance(ty, TupleType):
            return False
        if len(self.elements) != len(ty.elements):
            return False
        return all(traits.is_convertible(oth_el, el) for el, oth_el in zip(self.elements, ty.elements))

    def is_implicitly_constructible_from(self, ty: Type) -> bool:
        if not isinstance(ty, TupleType):
            return False
        if len(self.elements) != len(ty.elements):
            return False
        return all(traits.is_implicitly_convertible(oth_el, el) for el, oth_el in zip(self.elements, ty.elements))


@dataclasses.dataclass
class StructType(Type):
    """A usual struct type that consists of an ordered list of named fields."""

    fields: list[tuple[str, Type]]
    """
    The names and types of the fields of the struct. Ordering of the elements
    matters.
    """

    def __init__(self, fields: list[tuple[str, Type]]):
        self.fields = fields

    def __str__(self):
        elements = ", ".join(f"{field[0]}: {str(field[1])}" for field in self.fields)
        return f"struct[{elements}]"

    def __eq__(self, other):
        if isinstance(other, StructType):
            return self.fields == other.fields
        return False


@dataclasses.dataclass(frozen=True)
class FunctionParameter:
    """Represents a function parameter within callable types."""

    ty: Type
    """The type of the function parameter."""

    name: str
    """The name of the function parameter."""

    positional: bool
    """
    Whether the corresponding argument can be supplied as a positional in a
    function call.
    """

    keyword: bool
    """
    Whether the corresponding argument can be supplied as a keyword argument
    in a function call.
    """


@dataclasses.dataclass
class FunctionType(Type, traits.CallableTrait):
    """A usual function type that defines the parameters and the return type."""

    parameters: list[FunctionParameter]
    """
    The parameters of the function. For positional arguments, the position is
    the same as the index in this parameter list.
    """

    result: Optional[Type]
    """
    The return type of this function. Result is omitted (i.e. None) for a
    function that returns nothing (i.e. void). 
    """

    def __init__(self, parameters: list[FunctionParameter], result: Optional[Type]):
        self.parameters = parameters
        self.result = result

    def __str__(self):
        return f"({', '.join([str(param.ty) for param in self.parameters])}) -> {self.result}"

    def is_callable(self, args: Sequence[FunctionArgument]) -> traits.CallValidity:
        from gt4py.next.new_type_system import utils
        try:
            assigned = utils.assign_arguments(self.parameters, args)
            for param, arg in zip(self.parameters, assigned):
                if not traits.is_implicitly_convertible(arg.ty, param.ty):
                    return traits.CallValidity(
                        [
                            (f"argument cannot be implicitly converted from '{arg.ty}' to '{param.ty}' "
                             f"for parameter '{param.name}'")
                        ]
                    )
            return traits.CallValidity(self.result)
        except ValueError as error:
            return traits.CallValidity([str(error)])


# -------------------------------------------------------------------------------
# Aliases
# -------------------------------------------------------------------------------

@dataclasses.dataclass
class BoolType(IntegerType):
    def __init__(self):
        super().__init__(1, False)


@dataclasses.dataclass
class Int8Type(IntegerType):
    def __init__(self):
        super().__init__(8, True)


@dataclasses.dataclass
class Int16Type(IntegerType):
    def __init__(self):
        super().__init__(16, True)


@dataclasses.dataclass
class Int32Type(IntegerType):
    def __init__(self):
        super().__init__(32, True)


@dataclasses.dataclass
class Int64Type(IntegerType):
    def __init__(self):
        super().__init__(64, True)


@dataclasses.dataclass
class Uint8Type(IntegerType):
    def __init__(self):
        super().__init__(8, False)


@dataclasses.dataclass
class Uint16Type(IntegerType):
    def __init__(self):
        super().__init__(16, False)


@dataclasses.dataclass
class Uint32Type(IntegerType):
    def __init__(self):
        super().__init__(32, False)


@dataclasses.dataclass
class Uint64Type(IntegerType):
    def __init__(self):
        super().__init__(64, False)


@dataclasses.dataclass
class Float16Type(FloatType):
    def __init__(self):
        super().__init__(16)


@dataclasses.dataclass
class Float32Type(FloatType):
    def __init__(self):
        super().__init__(32)


@dataclasses.dataclass
class Float64Type(FloatType):
    def __init__(self):
        super().__init__(64)
