import enum
from typing import List, Union

from eve import Node
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import StrEnum, SymbolRef
from functional.iterator.util.sym_validation import validate_symbol_refs


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


class TemplatedFunCall(Expr):
    fun: Expr  # VType[Callable]
    template_args: List[Expr]
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


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    function_definitions: List[FunctionDefinition]
    executions: List[StencilExecution]
    offset_declarations: List[str]
    grid_type: GridType

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "deref",
            "shift",
            "tuple",
            "get",
            "can_deref",
            "domain",  # TODO(havogt) decide if domain is part of IR
            "named_range",
        ]
    )

    _validate_symbol_refs = validate_symbol_refs()
