import enum
from typing import ClassVar, List, Union

import eve
from eve.traits import SymbolName, SymbolTableTrait, ValidatedSymbolTableTrait
from eve.type_definitions import StrEnum, SymbolRef


@eve.utils.noninstantiable
class Node(eve.Node):
    pass


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


class Sym(Node):  # helper
    id: SymbolName  # noqa: A003


class Expr(Node):
    ...


class UnaryExpr(Expr):
    op: str
    expr: Expr


class BinaryExpr(Expr):
    op: str
    lhs: Expr
    rhs: Expr


class TernaryExpr(Expr):
    cond: Expr
    true_expr: Expr
    false_expr: Expr


class Literal(Expr):
    value: str
    type: str  # noqa: A003


class OffsetLiteral(Expr):
    value: Union[int, str]


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


class Backend(Node):
    domain: Union[SymRef, FunCall]  # TODO(havogt) `FunCall` only if domain will be part of the IR


class StencilExecution(Node):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: SymRef
    inputs: List[SymRef]


BUILTINS = {
    "deref",
    "shift",
    "make_tuple",
    "tuple_get",
    "can_deref",
    "domain",  # TODO(havogt) decide if domain is part of IR
    "named_range",
}


class FencilDefinition(Node, ValidatedSymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    function_definitions: List[FunctionDefinition]
    executions: List[StencilExecution]
    offset_declarations: List[str]
    grid_type: GridType

    _NODE_SYMBOLS_: ClassVar = [Sym(id=name) for name in BUILTINS]
