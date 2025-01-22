# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from gt4py.eve import Coerced, SymbolName, datamodels
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.next import common
from gt4py.next.iterator import builtins
from gt4py.next.program_processors.codegens.gtfn.gtfn_im_ir import ImperativeFunctionDefinition
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Node, Sym, SymRef


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


class CastExpr(Expr):
    obj_expr: Expr
    new_dtype: SymRef


class Literal(Expr):
    value: str
    type: str


class IntegralConstant(Expr):
    value: int  # generalize to other types if needed


class OffsetLiteral(Expr):
    value: Union[int, str]


class Lambda(Expr, SymbolTableTrait):
    params: list[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: list[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: list[Sym]
    expr: Expr


class ScanPassDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: list[Sym]
    expr: Expr
    forward: bool


class TaggedValues(Node):
    tags: list[Expr]
    values: list[Expr]


class CartesianDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues


class UnstructuredDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues
    connectivities: list[SymRef]  # SymRef to offset declaration


class Backend(Node):
    domain: Union[SymRef, CartesianDomain, UnstructuredDomain]


def _is_tuple_expr_of(pred: Callable[[Expr], bool], expr: Expr) -> bool:
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "tuple_get"
        and len(expr.args) == 2
        and _is_tuple_expr_of(pred, expr.args[1])
    ):
        return True
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "make_tuple"
        and all(_is_tuple_expr_of(pred, arg) for arg in expr.args)
    ):
        return True
    return pred(expr)


class SidComposite(Expr):
    values: list[Expr]

    @datamodels.validator("values")
    def _values_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: list[Expr]
    ) -> None:
        if not all(
            isinstance(el, (SidFromScalar, SidComposite))
            or _is_tuple_expr_of(lambda expr: isinstance(expr, (SymRef, Literal)), el)
            for el in value
        ):
            raise ValueError(
                "Only 'SymRef', 'Literal', tuple expr thereof, 'SidFromScalar', or 'SidComposite' allowed."
            )


def _might_be_scalar_expr(expr: Expr) -> bool:
    if isinstance(expr, BinaryExpr):
        return all(_is_tuple_expr_of(_might_be_scalar_expr, arg) for arg in (expr.lhs, expr.rhs))
    if isinstance(expr, UnaryExpr):
        return _is_tuple_expr_of(_might_be_scalar_expr, expr.expr)
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id in ARITHMETIC_BUILTINS
    ):
        return all(_might_be_scalar_expr(arg) for arg in expr.args)
    if isinstance(expr, CastExpr):
        return _might_be_scalar_expr(expr.obj_expr)
    if _is_tuple_expr_of(lambda e: isinstance(e, (SymRef, Literal)), expr):
        return True
    return False


class SidFromScalar(Expr):
    arg: Expr

    @datamodels.validator("arg")
    def _arg_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: Expr
    ) -> None:
        if not _might_be_scalar_expr(value):
            raise ValueError(
                "Only 'SymRef', 'Literal', arithmetic op or tuple expr thereof allowed."
            )


class Stmt(Node):
    pass


class StencilExecution(Stmt):
    backend: Backend
    stencil: SymRef
    output: Union[SymRef, SidComposite]
    inputs: list[Union[SymRef, SidComposite, SidFromScalar, FunCall]]

    @datamodels.validator("inputs")
    def _arg_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, inputs: list[Expr]
    ) -> None:
        for inp in inputs:
            if not _is_tuple_expr_of(
                lambda expr: isinstance(expr, (SymRef, SidComposite, SidFromScalar))
                or (
                    isinstance(expr, FunCall)
                    and isinstance(expr.fun, SymRef)
                    and expr.fun.id == "index"
                ),
                inp,
            ):
                raise ValueError(
                    "Only 'SymRef', 'SidComposite', 'SidFromScalar', 'index' call or tuple expr thereof allowed."
                )


class Scan(Node):
    function: SymRef
    output: int
    inputs: list[int]
    init: Expr


class ScanExecution(Stmt):
    backend: Backend
    scans: list[Scan]
    args: list[Expr]
    axis: SymRef


class IfStmt(Stmt):
    cond: Expr
    true_branch: list[Stmt]
    false_branch: list[Stmt]


class TemporaryAllocation(Node):
    id: SymbolName
    dtype: str
    domain: Union[SymRef, CartesianDomain, UnstructuredDomain]


GTFN_BUILTINS = [
    "deref",
    "shift",
    "make_tuple",
    "tuple_get",
    "can_deref",
    "cartesian_domain",
    "unstructured_domain",
    "named_range",
    "reduce",
    "index",
]
ARITHMETIC_BUILTINS = builtins.ARITHMETIC_BUILTINS
TYPEBUILTINS = builtins.TYPE_BUILTINS

BUILTINS = {*GTFN_BUILTINS, *ARITHMETIC_BUILTINS, *TYPEBUILTINS}


class TagDefinition(Node):
    name: Sym
    alias: Optional[Union[str, SymRef]] = None


class Program(Node, ValidatedSymbolTableTrait):
    id: SymbolName
    params: list[Sym]
    function_definitions: list[
        Union[FunctionDefinition, ScanPassDefinition, ImperativeFunctionDefinition]
    ]
    executions: list[Stmt]
    offset_definitions: list[TagDefinition]
    grid_type: common.GridType
    temporaries: list[TemporaryAllocation]

    _NODE_SYMBOLS_: ClassVar[list[Sym]] = [Sym(id=name) for name in BUILTINS]
