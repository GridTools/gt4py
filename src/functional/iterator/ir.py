from typing import List, Union

import eve
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from eve.utils import noninstantiable
from functional.iterator.util.sym_validation import validate_symbol_refs


@noninstantiable
class Node(eve.Node):
    def __str__(self):
        from functional.iterator.pretty_printer import pformat

        return pformat(self)


class Sym(Node):  # helper
    id: SymbolName  # noqa: A003


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


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    closures: List[StencilClosure]

    builtin_functions = [Sym(id=name) for name in BUILTINS]

    _validate_symbol_refs = validate_symbol_refs()
