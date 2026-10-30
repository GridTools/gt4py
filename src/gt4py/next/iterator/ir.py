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
    # TODO(havogt): Refactor to use declare Axis/Dimension at the Program level.
    # Now every use of the literal has to provide the kind, where usually we only care of the name.
    value: str
    kind: common.DimensionKind = common.DimensionKind.HORIZONTAL


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
    """
    Write `expr`, computed on `domain`, into `target`.

    For a tuple-valued `target` the `domain` is a matching tuple of domains (built with
    `make_tuple`), except that a single domain may also apply to an entire tuple subtree
    of the target, i.e. to each of its elements. This shallower structure is needed for
    variable-length tuple outputs, whose concrete length is unknown when the domain
    expression is constructed.
    """

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

    def __hash__(self) -> int:
        # Only here do types contribute to the hash as they are the starting point for
        # type inference, whereas in all other places the types can be recovered. Note
        # that equality still ignores types, so equal programs may have different hashes;
        # the users only consume the hash value itself, never using programs as dict keys.
        return hash((Node.__hash__(self), tuple(str(param.type) for param in self.params)))


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
SetAt.__hash__ = Node.__hash__  # type: ignore[method-assign]
IfStmt.__hash__ = Node.__hash__  # type: ignore[method-assign]
InfinityLiteral.__hash__ = Node.__hash__  # type: ignore[method-assign]
Temporary.__hash__ = Node.__hash__  # type: ignore[method-assign]
