from typing import List, Union

from eve import Node
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from functional.iterator.util.sym_validation import validate_symbol_refs


class Sym(Node):  # helper
    id: SymbolName  # noqa: A003


class Expr(Node):
    ...


class BoolLiteral(Expr):
    value: bool


class IntLiteral(Expr):
    value: int


class FloatLiteral(Expr):
    value: float  # TODO other float types


class StringLiteral(Expr):
    value: str


class NoneLiteral(Expr):
    _none_literal: int = 0


class OffsetLiteral(Expr):
    value: Union[int, str]

    def __hash__(self):
        return self.value.__hash__()


class AxisLiteral(Expr):
    value: str


class SymRef(Expr):
    id: SymbolRef  # noqa: A003


class Lambda(Expr, SymbolTableTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    expr: Expr

    def __eq__(self, other):
        return isinstance(other, FunctionDefinition) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class StencilClosure(Node):
    domain: Expr
    stencil: Expr
    output: SymRef  # we could consider Expr for cases like make_tuple(out0,out1)
    inputs: List[SymRef]


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    closures: List[StencilClosure]


class Program(Node, SymbolTableTrait):
    function_definitions: List[FunctionDefinition]
    fencil_definitions: List[FencilDefinition]

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "domain",
            "named_range",
            "lift",
            "make_tuple",
            "tuple_get",
            "reduce",
            "deref",
            "shift",
            "scan",
            "plus",
            "minus",
            "multiplies",
            "divides",
            "eq",
            "less",
            "greater",
            "if_",
            "not_",
            "and_",
            "or_",
        ]
    )
    _validate_symbol_refs = validate_symbol_refs()
