# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import List, Set

import factory

from eve import concepts, visitors
from gtc import common, oir


class CartesianOffsetFactory(factory.Factory):
    class Meta:
        model = common.CartesianOffset

    i = 0
    j = 0
    k = 0


class FieldAccessFactory(factory.Factory):
    class Meta:
        model = oir.FieldAccess

    name = factory.Faker("word")
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = common.DataType.FLOAT32


class AssignStmtFactory(factory.Factory):
    class Meta:
        model = oir.AssignStmt

    left = factory.SubFactory(FieldAccessFactory)
    right = factory.SubFactory(FieldAccessFactory)


class TemporaryFactory(factory.Factory):
    class Meta:
        model = oir.Temporary

    name = factory.Faker("word")
    dtype = common.DataType.FLOAT32


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = oir.FieldDecl

    name = factory.Faker("word")
    dtype = common.DataType.FLOAT32


class HorizontalExecutionFactory(factory.Factory):
    class Meta:
        model = oir.HorizontalExecution

    body = factory.List([factory.SubFactory(AssignStmtFactory)])
    mask = None
    declarations: List[oir.LocalScalar] = []


class IntervalFactory(factory.Factory):
    class Meta:
        model = oir.Interval

    start = common.AxisBound.start()
    end = common.AxisBound.end()


class VerticalLoopFactory(factory.Factory):
    class Meta:
        model = oir.VerticalLoop

    interval = factory.SubFactory(IntervalFactory)
    horizontal_executions = factory.List([factory.SubFactory(HorizontalExecutionFactory)])
    loop_order = common.LoopOrder.PARALLEL
    declarations: List[oir.Temporary] = []


class StencilFactory(factory.Factory):
    class Meta:
        model = oir.Stencil

    name = factory.Faker("word")
    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])

    @factory.lazy_attribute
    def params(self):
        """Automatically collect undefined symbols and put them into the parameter list."""

        class CollectSymbolsAndRefs(visitors.NodeVisitor):
            def visit_Node(self, node: concepts.Node, *, symbols: Set[str], refs: Set[str]) -> None:
                for name, metadata in node.__node_children__.items():
                    type_ = metadata["definition"].type_
                    if isinstance(type_, type):
                        if issubclass(type_, oir.SymbolName):
                            symbols.add(getattr(node, name))
                        elif issubclass(type_, common.SymbolRef):
                            refs.add(getattr(node, name))
                self.generic_visit(node, symbols=symbols, refs=refs)

        symbols: Set[str] = set()
        refs: Set[str] = set()
        CollectSymbolsAndRefs().visit(self.vertical_loops, symbols=symbols, refs=refs)
        return [FieldDeclFactory(name=name) for name in refs - symbols]
