# GT4Py - GridTools Framework
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

import gt4py.eve as eve
from gt4py.eve import Coerced, SymbolName, SymbolRef, datamodels
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.eve.utils import noninstantiable


@noninstantiable
class Node(eve.Node):
    def __str__(self) -> str:
        from gt4py.next.iterator.pretty_printer import pformat

        return pformat(self)

    def __hash__(self) -> int:
        return hash(type(self)) ^ hash(
            tuple(
                hash(tuple(v)) if isinstance(v, list) else hash(v)
                for v in self.iter_children_values()
            )
        )


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
    domain: FunCall
    stencil: Expr
    output: Union[SymRef, FunCall]
    inputs: List[SymRef]

    @datamodels.validator("output")
    def _output_validator(self: datamodels.DataModelTP, attribute: datamodels.Attribute, value):
        if isinstance(value, FunCall) and value.fun != SymRef(id="make_tuple"):
            raise ValueError("Only FunCall to `make_tuple` allowed.")


UNARY_MATH_NUMBER_BUILTINS = {"abs", "not_"}
UNARY_MATH_FP_BUILTINS = {
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sqrt",
    "exp",
    "log",
    "gamma",
    "cbrt",
    "floor",
    "ceil",
    "trunc",
}
UNARY_MATH_FP_PREDICATE_BUILTINS = {"isfinite", "isinf", "isnan"}
BINARY_MATH_NUMBER_BUILTINS = {
    "minimum",
    "maximum",
    "fmod",
    "power",
    "plus",
    "minus",
    "multiplies",
    "divides",
    "mod",
    "eq",
    "less",
    "greater",
    "greater_equal",
    "less_equal",
    "not_eq",
    "floordiv",  # TODO not treated in gtfn?
    "and_",
    "or_",
    "xor_",
}


TYPEBUILTINS = {"int", "int32", "int64", "float", "float32", "float64", "bool"}

BUILTINS = {
    "cartesian_domain",
    "unstructured_domain",
    "named_range",
    "lift",
    "make_tuple",
    "tuple_get",
    "reduce",
    "deref",
    "can_deref",
    "shift",
    "ignore_shift",
    "translate_shift",
    "scan",
    "if_",
    "cast_",
    *UNARY_MATH_NUMBER_BUILTINS,
    *UNARY_MATH_FP_BUILTINS,
    *UNARY_MATH_FP_PREDICATE_BUILTINS,
    *BINARY_MATH_NUMBER_BUILTINS,
    *TYPEBUILTINS,
}


class FencilDefinition(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    closures: List[StencilClosure]

    _NODE_SYMBOLS_: ClassVar[List[Sym]] = [Sym(id=name) for name in BUILTINS]


# TODO(fthaler): just use hashable types in nodes (tuples instead of lists)
Sym.__hash__ = Node.__hash__  # type: ignore[assignment]
Expr.__hash__ = Node.__hash__  # type: ignore[assignment]
Literal.__hash__ = Node.__hash__  # type: ignore[assignment]
NoneLiteral.__hash__ = Node.__hash__  # type: ignore[assignment]
OffsetLiteral.__hash__ = Node.__hash__  # type: ignore[assignment]
AxisLiteral.__hash__ = Node.__hash__  # type: ignore[assignment]
SymRef.__hash__ = Node.__hash__  # type: ignore[assignment]
Lambda.__hash__ = Node.__hash__  # type: ignore[assignment]
FunCall.__hash__ = Node.__hash__  # type: ignore[assignment]
FunctionDefinition.__hash__ = Node.__hash__  # type: ignore[assignment]
StencilClosure.__hash__ = Node.__hash__  # type: ignore[assignment]
FencilDefinition.__hash__ = Node.__hash__  # type: ignore[assignment]
