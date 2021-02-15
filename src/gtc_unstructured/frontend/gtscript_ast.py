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
from typing import List, Union

import gtc_unstructured.irs.common as common
from eve import Node


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


class Symbol(Expr):
    name: str


class IterationOrder(GTScriptASTNode):
    order: str


# TODO(tehrengruber): use type parameter see https://github.com/samuelcolvin/pydantic/pull/595
# T = TypeVar('T')
# class Constant(GTScriptASTNode, Generic[T]):
#    value: T


class Constant(Expr):
    # TODO(tehrengruber): use StrictStr, StrictInt, StrictFloat as pydantic automatically converts into the first
    #  type it occurs. As a result currently all integers become floats
    value: Union[float, int, None, str]


class Interval(GTScriptASTNode):
    """
    Interval [start, stop[
    """

    start: Constant  # TODO(tehrengruber): use Constant[Union[int, str, None]]
    stop: Constant


# TODO(tehrengruber): allow interval(...) by introducing Optional(captures={...}) placeholder
# Optional(captures={start=0, end=None})


class LocationSpecification(GTScriptASTNode):
    name: Symbol
    location_type: str


# TODO(tehrengruber): proper cannonicalization (CanBeCanonicalizedTo[Subscript] ?)
class SubscriptSingle(Expr):
    value: Symbol
    index: str


class SubscriptMultiple(Expr):
    value: Symbol
    indices: List[Union[Symbol, SubscriptSingle, "SubscriptMultiple"]]


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class Call(Expr):
    args: List[Expr]
    func: str


# TODO(tehrengruber): can be enabled as soon as eve_toolchain#58 lands
# class Call(Generic[T]):
#    name: str
#    return_type: T
#    arg_types: Ts
#    args: List[Expr]


class LocationComprehension(GTScriptASTNode):
    target: Symbol
    iterator: Call


class Generator(Expr):
    generators: List[LocationComprehension]
    elt: Expr


class Assign(Statement):
    target: Union[Symbol, SubscriptSingle, SubscriptMultiple]
    value: Expr


class Stencil(GTScriptASTNode):
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
    name: str
    type_: Union[Symbol, Union[SubscriptMultiple, SubscriptSingle]]
    # is_keyword: bool


class Computation(GTScriptASTNode):
    name: str
    arguments: List[Argument]
    stencils: List[Stencil]
    # TODO(tehrengruber): use the following as soon as nodes support type parameters:
    #  stencils: List[Union[Stencil[Stencil[Statement]], Stencil[Statement]]]
