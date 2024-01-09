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
    width: int
    signed: bool

    def __init__(self, width: int, signed: bool):
        assert (width in [1, 8, 16, 32, 64])
        if width == 1:
            assert (signed == False)
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
    width: int

    def __init__(self, width: int):
        assert (width in [16, 32, 64])
        self.width = width

    def __str__(self):
        return f"float{self.width}"

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
    elements: list[Type]

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
    fields: list[tuple[str, Type]]

    def __init__(self, fields: list[tuple[str, Type]]):
        self.fields = fields

    def __str__(self):
        elements = ", ".join(f"{field[0]}: {str(field[1])}" for field in self.fields)
        return f"struct[{elements}]"


@dataclasses.dataclass(frozen=True)
class FunctionParameter:
    ty: Type
    name: str
    positional: bool
    keyword: bool


@dataclasses.dataclass
class FunctionType(Type, traits.CallableTrait):
    parameters: list[FunctionParameter]
    result: Optional[Type]

    def __init__(self, parameters: list[FunctionParameter], result: Optional[Type]):
        self.parameters = parameters
        self.result = result

    def is_callable(self, args: Sequence[FunctionArgument]) -> tuple[bool, str | Any]:
        positionals = [param for param in self.parameters if param.positional]
        keywords = {param.name: param for param in self.parameters if param.keyword}
        supplied = {param.name: False for param in self.parameters}
        for arg in args:
            if isinstance(arg.location, int):
                if arg.location >= len(positionals):
                    return False, "too many positional arguments supplied"
                param = positionals[arg.location]
            else:
                if arg.location not in keywords:
                    return False, f"unexpected keyword argument '{arg.location}'"
                param = keywords[arg.location]
            if supplied[param.name]:
                return False, f"argument '{param.name}' supplied multiple times"
            if not traits.is_implicitly_convertible(arg.ty, param.ty):
                return False, f"argument cannot be implicitly converted from '{arg.ty}' to '{param.ty}'  for parameter '{param.name}'"
            supplied[param.name] = True
        for param_name, p_supplied in supplied.items():
            if not p_supplied:
                return False, f"argument '{param_name}' missing"
        return True, self.result


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
