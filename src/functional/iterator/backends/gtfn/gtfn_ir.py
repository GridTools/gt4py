from typing import List, Union

from eve import Node
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from functional.iterator.util.sym_validation import validate_symbol_refs


class Sym(Node):  # helper
    id: SymbolName  # noqa: A003


class Expr(Node):
    ...


class UnaryExpr(Expr):
    op: str
    expr: Expr


class BoolLiteral(Expr):
    value: bool


class IntLiteral(Expr):
    value: int


class FloatLiteral(Expr):
    value: float  # TODO other float types


class StringLiteral(Expr):
    value: str


class OffsetLiteral(Expr):
    # Takes into account the change to offset=(tag,value)
    # TODO upstream the change
    tag: str
    value: int


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


class Backend(Node):
    domain: SymRef
    backend_tag: str


class StencilExecution(Node):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: SymRef
    inputs: List[SymRef]


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    executions: List[StencilExecution]


class Program(Node, SymbolTableTrait):
    function_definitions: List[FunctionDefinition]
    fencil_definitions: List[FencilDefinition]

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "deref",
        ]
    )

    _validate_symbol_refs = validate_symbol_refs()
