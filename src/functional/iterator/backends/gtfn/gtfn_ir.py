import enum
from typing import Union

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
    params: list[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: list[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    expr: Expr


class ScanPassDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    expr: Expr
    forward: bool


class Backend(Node):
    domain: Union[SymRef, FunCall]  # TODO(havogt) `FunCall` only if domain will be part of the IR


class StencilExecution(Node):
    backend: Backend
    stencil: SymRef
    output: SymRef
    inputs: list[SymRef]


class Scan(Node):
    function: SymRef
    output: Literal
    inputs: list[Literal]
    init: Expr


class ScanExecution(Node):
    backend: Backend
    scans: list[Scan]
    args: list[SymRef]


class TemporaryAllocation(Node):
    id: SymbolName  # noqa: A003
    dtype: str
    # TODO: domain: ??


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: list[Sym]
    function_definitions: list[Union[FunctionDefinition, ScanPassDefinition]]
    executions: list[Union[StencilExecution, ScanExecution]]
    offset_declarations: list[str]
    grid_type: GridType
    temporaries: list[TemporaryAllocation]

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "deref",
            "shift",
            "make_tuple",
            "tuple_get",
            "can_deref",
            "domain",  # TODO(havogt) decide if domain is part of IR
            "named_range",
        ]
    )

    _validate_symbol_refs = validate_symbol_refs()
