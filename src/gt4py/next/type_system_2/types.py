from . import traits

class Type:
    def implements(self, trait: traits.Trait):
        return isinstance(self, traits.Trait) and trait.satisfied_by(self)


class IntegerType(Type, traits.SignednessTrait):
    width: int
    signed: bool

    def __init__(self, width: int, signed: bool):
        Type.__init__(self)
        traits.SignednessTrait.__init__(self, signed)
        assert(width in [1, 8, 16, 32, 64])
        if width == 1:
            assert(signed == False)
        self.width = width
        self.signed = signed


class FloatType(Type, traits.SignednessTrait):
    width: int

    def __init__(self, width: int):
        traits.SignednessTrait.__init__(self, True)
        assert (width in [16, 32, 64])
        self.width = width
        traits.SignednessTrait(True)


class TupleType(Type):
    elements: list[Type]

    def __init__(self, elements: list[Type]):
        self.elements = elements


class StructType(Type):
    fields: list[tuple[str, Type]]

    def __init__(self, fields: list[tuple[str, Type]]):
        self.fields = fields



