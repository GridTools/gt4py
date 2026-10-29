# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import typing
from typing import TYPE_CHECKING, ClassVar, List, Optional, Union

import gt4py.eve as eve
from gt4py.eve import Coerced, SymbolName, SymbolRef, concepts, utils as eve_utils
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.next import common, fingerprinting
from gt4py.next.iterator.builtins import BUILTINS
from gt4py.next.type_system import type_specifications as ts


DimensionKind = common.DimensionKind

# Generate an unique fingerprint for `eve.Node`s ignoring the "location" and "type" attribute.
# TODO(tehrengruber): this is a workaround for the fact that `eve.Node`s type can be
# set after creation, in the type inference passes.
lenient_ir_fingerprinter: fingerprinting.Fingerprinter = fingerprinting.make_fingerprinter(
    deconstructor=fingerprinting.make_lenient_data_deconstructor(
        {concepts.Node: fingerprinting.skipping_fields_node_deconstructor("location", "type")}
    )
)


@eve_utils.noninstantiable
class Node(eve.Node):
    location: Optional[concepts.SourceLocation] = eve.field(default=None, repr=False, compare=False)

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


@eve_utils.noninstantiable
class Expr(Node): ...


class Literal(Expr):
    value: str
    type: ts.ScalarType


class InfinityLiteral(Expr):
    # TODO(tehrengruber): self referential `ClassVar` not supported in eve.
    if TYPE_CHECKING:
        POSITIVE: ClassVar[InfinityLiteral]
        NEGATIVE: ClassVar[InfinityLiteral]

    name: typing.Literal["POSITIVE", "NEGATIVE"]

    def __str__(self):
        return f"{type(self).__name__}.{self.name}"


InfinityLiteral.NEGATIVE = InfinityLiteral(name="NEGATIVE")
InfinityLiteral.POSITIVE = InfinityLiteral(name="POSITIVE")


# TODO(tehrengruber): allow int only and create OffsetRef for str instead
class OffsetLiteral(Expr):
    value: Union[int, str]


class AxisLiteral(Expr):
    #: The dimension, as a self-contained value: kind, category and staggered base are in the
    #: IR, so nothing below the frontend imports the module that declared it. A dimension class
    #: passed to the constructor is converted.
    dim: common.DimensionRef = eve.field(converter=common.dimension_ref, kw_only=True)

    @property
    def value(self) -> common.Tag:
        """The dimension's tag, its IR spelling."""
        return self.dim.tag

    @property
    def kind(self) -> common.DimensionKind:
        return self.dim.kind


class CartesianOffset(Expr):
    domain: AxisLiteral
    codomain: AxisLiteral


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

    _NODE_SYMBOLS_: ClassVar[List[Sym]] = [
        Sym(id=name) for name in sorted(BUILTINS)
    ]  # sorted for serialization stability


# TODO(fthaler): just use hashable types in nodes (tuples instead of lists)
Sym.__hash__ = Node.__hash__  # type: ignore[method-assign]
Expr.__hash__ = Node.__hash__  # type: ignore[method-assign]
Literal.__hash__ = Node.__hash__  # type: ignore[method-assign]
OffsetLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
AxisLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
CartesianOffset.__hash__ = Node.__hash__  # type: ignore[method-assign]
SymRef.__hash__ = Node.__hash__  # type: ignore[method-assign]
Lambda.__hash__ = Node.__hash__  # type: ignore[method-assign]
FunCall.__hash__ = Node.__hash__  # type: ignore[method-assign]
FunctionDefinition.__hash__ = Node.__hash__  # type: ignore[method-assign]
Program.__hash__ = Node.__hash__  # type: ignore[method-assign]
SetAt.__hash__ = Node.__hash__  # type: ignore[method-assign]
IfStmt.__hash__ = Node.__hash__  # type: ignore[method-assign]
InfinityLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
Temporary.__hash__ = Node.__hash__  # type: ignore[method-assign]
