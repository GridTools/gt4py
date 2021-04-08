# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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
# todo(tehrengruber): document nodes
from typing import Any, Dict, List, Optional, Type, Union

import pydantic
from pydantic import StrictFloat, StrictInt, StrictStr

import eve
import eve.type_definitions
import gtc_unstructured.irs.common as common
from eve import Node, Str, StrEnum, SymbolTableTrait
from eve.type_definitions import SymbolName
from gtc import common as stable_gtc_common

from . import built_in_types
from .built_in_functions import BuiltInFunction
from .built_in_types import BuiltInTypeMeta


__all__ = [
    "GTScriptASTNode",
    "Statement",
    "Expr",
    "Symbol",
    "IterationOrder",
    "Constant",
    "Interval",
    "LocationSpecification",
    "SubscriptSingle",
    "SubscriptMultiple",
    "BinaryOp",
    "Call",
    "LocationComprehension",
    "Generator",
    "Assign",
    "Stencil",
    "Pass",
    "Argument",
    "Computation",
]


class GTScriptASTNode(Node):
    pass


class Statement(GTScriptASTNode):
    pass


class Expr(GTScriptASTNode):
    pass


class SymbolRef(Expr):
    name: eve.type_definitions.SymbolRef
    type_: Optional[Any] = None


class IterationOrder(GTScriptASTNode):
    order: str


# TODO(tehrengruber): use type parameter see https://github.com/samuelcolvin/pydantic/pull/595
# T = TypeVar('T')
# class Constant(GTScriptASTNode, Generic[T]):
#    value: T


class Constant(Expr):
    value: Union[StrictStr, StrictInt, StrictFloat, None]


class Interval(GTScriptASTNode):
    """
    Interval [start, stop[
    """

    start: Constant  # TODO(tehrengruber): use Constant[Union[int, str, None]]
    stop: Constant


# TODO(tehrengruber): allow interval(...) by introducing Optional(captures={...}) placeholder
# Optional(captures={start=0, end=None})


class LocationSpecification(GTScriptASTNode):
    name: SymbolName
    location_type: SymbolRef


class Subscript(Expr):
    value: SymbolRef
    indices: List[Union[SymbolRef, "Subscript"]]


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class UnaryOp(Expr):
    op: stable_gtc_common.UnaryOperator
    operand: Expr


class Call(Expr):
    args: List[Expr]
    func: SymbolRef


class SubscriptCall(Expr):
    args: List[Expr]
    func: Subscript


class List_(Expr):
    elts: List[Expr]


# TODO(tehrengruber): can be enabled as soon as eve_toolchain#58 lands
# class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]


class LocationComprehension(GTScriptASTNode):
    target: SymbolName
    iterable: Subscript


class Generator(Expr, SymbolTableTrait):
    generators: List[LocationComprehension]
    elt: Expr


class Assign(Statement):
    target: Union[SymbolRef, Subscript]
    value: Expr


class Stencil(GTScriptASTNode, SymbolTableTrait):
    iteration_spec: List[Union[IterationOrder, LocationSpecification, Interval]]
    body: List[
        Union[Statement, "Stencil"]
    ]  # TODO(tehrengruber): stencil only allowed non-canonicalized


# TODO(tehrengruber): attributes are not supported yet. we need to decide on how to handle them first
# class Attribute(GT4PyAstNode):
#    attr: str
#    value: Union[Attribute, Name]
#
#    @staticmethod
#    def template():
#        return ast.Attribute(attr=Capture("attr"), value=Capture("value"))


class Pass(Statement):
    pass


class Argument(GTScriptASTNode):
    name: SymbolName
    type_: BuiltInTypeMeta
    # type_: Union[SymbolRef, Union[Subscript]]
    # is_keyword: bool


class External(GTScriptASTNode):
    name: SymbolName
    value: Union[
        StrictInt,
        StrictFloat,
        None,
        common.LocationType,
        common.DataType,
        BuiltInTypeMeta,
        BuiltInFunction,
    ]

    class Config:
        arbitrary_types_allowed = True

    @property
    def type_(self):
        if issubclass(
            type(self.value), built_in_types.Type_
        ):  # for now type of a type is the type itself
            return self.value
        return type(self.value)


class TemporaryFieldDecl(GTScriptASTNode):
    name: SymbolName
    type_: BuiltInTypeMeta


class TemporarySparseFieldDecl(GTScriptASTNode):
    name: SymbolName
    type_: BuiltInTypeMeta


class Computation(GTScriptASTNode, SymbolTableTrait):
    # TODO(tehrengruber): use the following as soon as nodes support type parameters:
    #  stencils: List[Union[Stencil[Stencil[Statement]], Stencil[Statement]]]
    declarations: List[
        Union[TemporaryFieldDecl, TemporarySparseFieldDecl]
    ] = []  # derived attribute

    name: str
    arguments: List[Argument]
    stencils: List[Stencil]
    externals: List[External]

    # @pydantic.root_validator(skip_on_failure=True)
    # def _type_propagate(  # type: ignore  # validators are classmethods
    #        cls: Type["Computation"], values: Dict[str, Any]
    # ) -> Dict[str, Any]:
    #    return TypePropagator.apply(values["symtable_"], values)

    @pydantic.root_validator(skip_on_failure=True)
    def _implicit_connectivity_type_decls(  # type: ignore  # validators are classmethods
        cls: Type["Computation"], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        # TODO(tehrengruber): hacky
        for arg in values["arguments"]:
            if issubclass(arg.type_, built_in_types.Connectivity):
                values["externals"].append(External(name=arg.type_.class_name, value=arg.type_))
        del values["symtable_"]
        values["symtable_"] = cls._collect_symbols(values)
        return values

    @pydantic.root_validator(skip_on_failure=True)
    def _extract_temporary_fields(  # type: ignore  # validators are classmethods
        cls: Type["Computation"], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        from .passes.temporary_field_decl_extractor import TemporaryFieldDeclExtractor

        values["declarations"] = values["declarations"] + TemporaryFieldDeclExtractor.apply(
            values["symtable_"], values["stencils"]
        )
        # TODO Find a solution for this pattern:
        # - During transformation from python ast to gtscript_ast, we cannot easily create Symbols for temporaries
        #   (because the pass is generic `transform()` in `py_to_gtscript.py`).
        # - If we want to use Symbol-SymbolRef check, we should represent the Temporary as a Symbol.
        # - The first place where we can easily do this is here.
        # - Considerations:
        #   - The automtic symbol collection (vis SymbolTableTrait) already run (and we actually want to use it in the pass),
        #     but we need to run again afterwards. Ugly... If we at some point automatically run the SymbolRef check
        #     in this root node, the pattern will fail as we cannot define run-in-between-collection-and-ref-check.
        #   - The temporaries in `declarations` are like cached properties, but not really as they declare Symbols.
        #   - If there wasn't the Symbol declarations, we could run this in the lowering to GTIR.
        del values["symtable_"]
        values["symtable_"] = cls._collect_symbols(values)
        return values

    # TODO: fix. currently fails as the symbol table is not built up when the stencils field is visited
    # _validate_symbol_refs = stable_gtc_common.validate_symbol_refs()
