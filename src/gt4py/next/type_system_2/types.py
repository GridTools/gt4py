from . import traits
import dataclasses
from typing import Optional


class Type:
    def implements(self, trait: traits.Trait):
        return isinstance(self, traits.Trait) and trait.satisfied_by(self)


class IntegerType(Type, traits.SignednessTrait, traits.FromTrait):
    width: int
    signed: bool

    def __init__(self, width: int, signed: bool):
        Type.__init__(self)
        traits.SignednessTrait.__init__(self, signed)
        traits.FromTrait.__init__(self, self, self.is_convertible_from)
        assert(width in [1, 8, 16, 32, 64])
        if width == 1:
            assert(signed == False)
        self.width = width
        self.signed = signed

    def is_convertible_from(self, other: Type) -> bool:
        return isinstance(other, IntegerType) or isinstance(other, FloatType)


class FloatType(Type, traits.SignednessTrait, traits.FromTrait):
    width: int

    def __init__(self, width: int):
        traits.SignednessTrait.__init__(self, True)
        traits.FromTrait.__init__(self, self, self.is_convertible_from)
        assert (width in [16, 32, 64])
        self.width = width
        traits.SignednessTrait(True)

    def is_convertible_from(self, other: Type) -> bool:
        return isinstance(other, IntegerType) or isinstance(other, FloatType)


class TupleType(Type):
    elements: list[Type]

    def __init__(self, elements: list[Type]):
        self.elements = elements


class StructType(Type):
    fields: list[tuple[str, Type]]

    def __init__(self, fields: list[tuple[str, Type]]):
        self.fields = fields


@dataclasses.dataclass
class FunctionArgument:
    ty: Type
    name: str
    positional: bool
    keyword: bool


class FunctionType(Type):
    arguments: list[FunctionArgument]
    result: Optional[Type]

    def __init__(self, arguments: list[FunctionArgument], result: Optional[Type]):
        self.arguments = arguments
        self.result = result
