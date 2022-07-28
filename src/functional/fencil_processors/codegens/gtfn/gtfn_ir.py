# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from typing import ClassVar, List, Union

import eve
from eve import Coerced, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from functional import common
from functional.iterator import ir as itir


@eve.utils.noninstantiable
class Node(eve.Node):
    pass


class Sym(Node):  # helper
    id: Coerced[SymbolName]  # noqa: A003


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


class TaggedValues(Node):
    tags: List[Expr]
    values: List[Expr]


class CartesianDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues


class UnstructuredDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues
    connectivities: List[SymRef]  # SymRef to offset declaration


class Backend(Node):
    domain: Union[SymRef, CartesianDomain, UnstructuredDomain]


class SidComposite(Expr):
    values: List[Expr]  # SymRef or SidComposite (need recursive type)


class StencilExecution(Node):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: Expr  # SymRef or SidComposite
    inputs: List[SymRef]


UNARY_MATH_NUMBER_BUILTINS = itir.UNARY_MATH_NUMBER_BUILTINS
UNARY_MATH_FP_BUILTINS = itir.UNARY_MATH_FP_BUILTINS
UNARY_MATH_FP_PREDICATE_BUILTINS = itir.UNARY_MATH_FP_PREDICATE_BUILTINS
BINARY_MATH_NUMBER_BUILTINS = itir.BINARY_MATH_NUMBER_BUILTINS

BUILTINS = {
    "deref",
    "shift",
    "make_tuple",
    "tuple_get",
    "can_deref",
    "cartesian_domain",
    "unstructured_domain",
    "named_range",
    *UNARY_MATH_NUMBER_BUILTINS,
    *UNARY_MATH_FP_BUILTINS,
    *UNARY_MATH_FP_PREDICATE_BUILTINS,
    *BINARY_MATH_NUMBER_BUILTINS,
}


class FencilDefinition(Node, ValidatedSymbolTableTrait):
    id: SymbolName  # noqa: A003
    params: List[Sym]
    function_definitions: List[FunctionDefinition]
    executions: List[StencilExecution]
    offset_declarations: List[Sym]
    grid_type: common.GridType

    _NODE_SYMBOLS_: ClassVar = [Sym(id=name) for name in BUILTINS]
