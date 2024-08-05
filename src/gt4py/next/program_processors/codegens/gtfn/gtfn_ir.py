# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import ClassVar, Optional, Union

from gt4py.eve import Coerced, SymbolName, datamodels
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.next import common
from gt4py.next.iterator import ir as itir
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


def _is_ref_or_tuple_expr_of_ref(expr: Expr) -> bool:
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "tuple_get"
        and len(expr.args) == 2
        and _is_ref_or_tuple_expr_of_ref(expr.args[1])
    ):
        return True
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "make_tuple"
        and all(_is_ref_or_tuple_expr_of_ref(arg) for arg in expr.args)
    ):
        return True
    if isinstance(expr, SymRef):
        return True
    return False


class SidComposite(Expr):
    values: list[Expr]

    @datamodels.validator("values")
    def _values_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: list[Expr]
    ) -> None:
        if not all(
            isinstance(el, (SidFromScalar, SidComposite)) or _is_ref_or_tuple_expr_of_ref(el)
            for el in value
        ):
            raise ValueError(
                "Only 'SymRef', tuple expr of 'SymRef', 'SidFromScalar', or 'SidComposite' allowed."
            )


class SidFromScalar(Expr):
    arg: Expr

    @datamodels.validator("arg")
    def _arg_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: Expr
    ) -> None:
        if not _is_ref_or_tuple_expr_of_ref(value):
            raise ValueError("Only 'SymRef' or tuple expr of 'SymRef' allowed.")


class StencilExecution(Node):
    backend: Backend
    stencil: SymRef
    output: Union[SymRef, SidComposite]
    inputs: list[Union[SymRef, SidComposite, SidFromScalar]]


class Scan(Node):
    function: SymRef
    output: int
    inputs: list[int]
    init: Expr


class ScanExecution(Node):
    backend: Backend
    scans: list[Scan]
    args: list[Expr]
    axis: SymRef


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
]
ARITHMETIC_BUILTINS = itir.ARITHMETIC_BUILTINS
TYPEBUILTINS = itir.TYPEBUILTINS

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
    executions: list[Union[StencilExecution, ScanExecution]]
    offset_definitions: list[TagDefinition]
    grid_type: common.GridType
    temporaries: list[TemporaryAllocation]

    _NODE_SYMBOLS_: ClassVar[list[Sym]] = [Sym(id=name) for name in BUILTINS]
