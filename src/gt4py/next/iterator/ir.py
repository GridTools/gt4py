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

import typing
from typing import ClassVar, List, Optional, Union

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
    # TODO(tehrengruber): Revisit. Using strings is a workaround to avoid coupling with the
    #   type inference.
    kind: typing.Literal["Iterator", "Value", None] = None
    dtype: Optional[
        tuple[str, bool]
    ] = None  # format: name of primitive type, boolean indicating if it is a list

    @datamodels.validator("kind")
    def _kind_validator(self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: str):
        if value and value not in ["Iterator", "Value"]:
            raise ValueError(f"Invalid kind `{value}`, must be one of `Iterator`, `Value`.")

    @datamodels.validator("dtype")
    def _dtype_validator(self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: str):
        if value and value[0] not in TYPEBUILTINS:
            raise ValueError(
                f"Invalid dtype `{value}`, must be one of `{'`, `'.join(TYPEBUILTINS)}`."
            )


@noninstantiable
class Expr(Node):
    ...


class Literal(Expr):
    value: str
    type: str  # noqa: A003

    @datamodels.validator("type")
    def _type_validator(self: datamodels.DataModelTP, attribute: datamodels.Attribute, value):
        if value not in TYPEBUILTINS:
            raise ValueError(f"{value} is not a valid builtin type.")


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


UNARY_MATH_NUMBER_BUILTINS = {"abs"}
UNARY_LOGICAL_BUILTINS = {"not_"}
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
    "plus",
    "minus",
    "multiplies",
    "divides",
    "mod",
    "floordiv",  # TODO see https://github.com/GridTools/gt4py/issues/1136
}
BINARY_MATH_COMPARISON_BUILTINS = {
    "eq",
    "less",
    "greater",
    "greater_equal",
    "less_equal",
    "not_eq",
}
BINARY_LOGICAL_BUILTINS = {
    "and_",
    "or_",
    "xor_",
}

ARITHMETIC_BUILTINS = {
    *UNARY_MATH_NUMBER_BUILTINS,
    *UNARY_LOGICAL_BUILTINS,
    *UNARY_MATH_FP_BUILTINS,
    *UNARY_MATH_FP_PREDICATE_BUILTINS,
    *BINARY_MATH_NUMBER_BUILTINS,
    "power",
    *BINARY_MATH_COMPARISON_BUILTINS,
    *BINARY_LOGICAL_BUILTINS,
}

#: builtin / dtype used to construct integer indices, like domain bounds
INTEGER_INDEX_BUILTIN = "int32"
INTEGER_BUILTINS = {"int32", "int64"}
FLOATING_POINT_BUILTINS = {"float32", "float64"}
TYPEBUILTINS = {*INTEGER_BUILTINS, *FLOATING_POINT_BUILTINS, "bool"}

GRAMMAR_BUILTINS = {
    "cartesian_domain",
    "unstructured_domain",
    "make_tuple",
    "tuple_get",
    "shift",
    "neighbors",
    "cast_",
}

BUILTINS = {
    *GRAMMAR_BUILTINS,
    "named_range",
    "list_get",
    "map_",
    "make_const_list",
    "lift",
    "reduce",
    "deref",
    "can_deref",
    "scan",
    "if_",
    *ARITHMETIC_BUILTINS,
    *TYPEBUILTINS,
}


class FencilDefinition(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    closures: List[StencilClosure]

    _NODE_SYMBOLS_: ClassVar[List[Sym]] = [Sym(id=name) for name in BUILTINS]


# TODO(fthaler): just use hashable types in nodes (tuples instead of lists)
Sym.__hash__ = Node.__hash__  # type: ignore[method-assign]
Expr.__hash__ = Node.__hash__  # type: ignore[method-assign]
Literal.__hash__ = Node.__hash__  # type: ignore[method-assign]
NoneLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
OffsetLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
AxisLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
SymRef.__hash__ = Node.__hash__  # type: ignore[method-assign]
Lambda.__hash__ = Node.__hash__  # type: ignore[method-assign]
FunCall.__hash__ = Node.__hash__  # type: ignore[method-assign]
FunctionDefinition.__hash__ = Node.__hash__  # type: ignore[method-assign]
StencilClosure.__hash__ = Node.__hash__  # type: ignore[method-assign]
FencilDefinition.__hash__ = Node.__hash__  # type: ignore[method-assign]
