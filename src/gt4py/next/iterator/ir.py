# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar, List, Optional, Union

import gt4py.eve as eve
from gt4py.eve import Coerced, SymbolName, SymbolRef
from gt4py.eve.concepts import SourceLocation
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.eve.utils import noninstantiable
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts


DimensionKind = common.DimensionKind


@noninstantiable
class Node(eve.Node):
    location: Optional[SourceLocation] = eve.field(default=None, repr=False, compare=False)

    # TODO(tehrengruber): include in comparison if value is not None
    type: Optional[ts.TypeSpec] = eve.field(default=None, repr=False, compare=False)

    def __str__(self) -> str:
        from gt4py.next.iterator.pretty_printer import pformat

        return pformat(self)

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                *(
                    tuple(v) if isinstance(v, list) else v
                    for (k, v) in self.iter_children_items()
                    if k not in ["location", "type"]
                ),
            )
        )


class Sym(Node):  # helper
    id: Coerced[SymbolName]


@noninstantiable
class Expr(Node): ...


class Literal(Expr):
    value: str
    type: ts.ScalarType


class NoneLiteral(Expr):
    _none_literal: int = 0


class OffsetLiteral(Expr):
    value: Union[int, str]


class AxisLiteral(Expr):
    # TODO(havogt): Refactor to use declare Axis/Dimension at the Program level.
    # Now every use of the literal has to provide the kind, where usually we only care of the name.
    value: str
    kind: common.DimensionKind = common.DimensionKind.HORIZONTAL


class SymRef(Expr):
    id: Coerced[SymbolRef]


class Lambda(Expr, SymbolTableTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: List[Sym]
    expr: Expr


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
    "neg",
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
BINARY_MATH_COMPARISON_BUILTINS = {"eq", "less", "greater", "greater_equal", "less_equal", "not_eq"}
BINARY_LOGICAL_BUILTINS = {"and_", "or_", "xor_"}

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

BUILTINS = {
    "tuple_get",
    "cast_",
    "cartesian_domain",
    "unstructured_domain",
    "make_tuple",
    "shift",
    "neighbors",
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
    "index",  # `index(dim)` creates a dim-field that has the current index at each point
    "as_fieldop",  # `as_fieldop(stencil, domain)` creates field_operator from stencil (domain is optional, but for now required for embedded execution)
    *ARITHMETIC_BUILTINS,
    *TYPEBUILTINS,
}


class Stmt(Node): ...


class SetAt(Stmt):  # from JAX array.at[...].set()
    expr: Expr  # only `as_fieldop(stencil)(inp0, ...)` in first refactoring
    domain: Expr
    target: Expr  # `make_tuple` or SymRef


class IfStmt(Stmt):
    cond: Expr
    true_branch: list[Stmt]
    false_branch: list[Stmt]


class Temporary(Node):
    id: Coerced[eve.SymbolName]
    domain: Optional[Expr] = None
    dtype: Optional[ts.ScalarType | ts.TupleType] = None


class Program(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]
    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    declarations: List[Temporary]
    body: List[Stmt]
    implicit_domain: bool = False

    _NODE_SYMBOLS_: ClassVar[List[Sym]] = [
        Sym(id=name) for name in sorted(BUILTINS)
    ]  # sorted for serialization stability


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
Program.__hash__ = Node.__hash__  # type: ignore[method-assign]
SetAt.__hash__ = Node.__hash__  # type: ignore[method-assign]
IfStmt.__hash__ = Node.__hash__  # type: ignore[method-assign]
