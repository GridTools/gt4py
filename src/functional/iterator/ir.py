from typing import ClassVar, List, Union

import eve
from eve import Coerced, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from eve.utils import noninstantiable


@noninstantiable
class Node(eve.Node):
    def __str__(self) -> str:
        from functional.iterator.pretty_printer import pformat

        return pformat(self)


class Sym(Node):  # helper
    id: Coerced[SymbolName]  # noqa: A003


@noninstantiable
class Expr(Node):
    ...


class Literal(Expr):
    value: str
    type: str  # noqa: A003


class NoneLiteral(Expr):
    _none_literal: int = 0


class OffsetLiteral(Expr):
    value: Union[int, str]


class AxisLiteral(Expr):
    value: str


class SymRef(Expr):
    id: Coerced[SymbolRef]  # noqa: A003


class Lambda(Expr, SymbolTableTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: List[Sym]
    expr: Expr


class StencilClosure(Node):
    domain: Expr
    stencil: Expr
    output: SymRef  # we could consider Expr for cases like make_tuple(out0,out1)
    inputs: List[SymRef]


BUILTINS = {
    "domain",
    "named_range",
    "lift",
    "make_tuple",
    "tuple_get",
    "reduce",
    "deref",
    "can_deref",
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
}


class FencilDefinition(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    closures: List[StencilClosure]

    _NODE_SYMBOLS_: ClassVar = [Sym(id=name) for name in BUILTINS]
