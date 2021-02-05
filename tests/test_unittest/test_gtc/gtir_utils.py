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

from typing import Set

import factory

from eve import concepts, visitors
from gtc import common, gtir


class DummyExpr(gtir.Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: common.DataType
    kind: common.ExprKind


class DummyExprFactory(factory.Factory):
    class Meta:
        model = DummyExpr

    dtype = common.DataType.FLOAT32
    kind = factory.Faker("random_element", elements=common.ExprKind)


class LiteralFactory(factory.Factory):
    class Meta:
        model = gtir.Literal

    value = "42.0"
    dtype = common.DataType.FLOAT32


class CartesianOffsetFactory(factory.Factory):
    class Meta:
        model = common.CartesianOffset

    i = 0
    j = 0
    k = 0


class FieldAccessFactory(factory.Factory):
    class Meta:
        model = gtir.FieldAccess

    name = factory.Faker("word")
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = common.DataType.FLOAT32


class ScalarAccessFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarAccess

    name = factory.Faker("word")
    dtype = common.DataType.FLOAT32


class ParAssignStmtFactory(factory.Factory):
    class Meta:
        model = gtir.ParAssignStmt

    left = factory.SubFactory(FieldAccessFactory)
    right = factory.SubFactory(FieldAccessFactory)


class BinaryOpFactory(factory.Factory):
    class Meta:
        model = gtir.BinaryOp

    op = common.ArithmeticOperator.ADD
    left = factory.SubFactory(FieldAccessFactory)
    right = factory.SubFactory(FieldAccessFactory)


class BlockStmtFactory(factory.Factory):
    class Meta:
        model = gtir.BlockStmt

    body = []


class FieldIfStmtFactory(factory.Factory):
    class Meta:
        model = gtir.FieldIfStmt

    cond = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.BOOL)
    true_branch = factory.SubFactory(BlockStmtFactory)
    false_branch = None


class ScalarIfStmtFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarIfStmt

    cond = factory.SubFactory(ScalarAccessFactory, dtype=common.DataType.BOOL)
    true_branch = factory.SubFactory(BlockStmtFactory)
    false_branch = None


class IntervalFactory(factory.Factory):
    class Meta:
        model = gtir.Interval

    start = common.AxisBound.start()
    end = common.AxisBound.end()


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = gtir.FieldDecl

    name = factory.Faker("word")
    dtype = common.DataType.FLOAT32


class ScalarDeclFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarDecl

    name = factory.Faker("word")
    dtype = common.DataType.FLOAT32


class VerticalLoopFactory(factory.Factory):
    class Meta:
        model = gtir.VerticalLoop

    interval = factory.SubFactory(IntervalFactory)
    loop_order = common.LoopOrder.PARALLEL
    temporaries = []
    body = factory.List([factory.SubFactory(ParAssignStmtFactory)])


class StencilFactory(factory.Factory):
    class Meta:
        model = gtir.Stencil

    name = factory.Faker("word")
    params = []
    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])

    @factory.lazy_attribute
    def params(self):
        """Automatically collect undefined symbols and put them into the parameter list."""

        class CollectSymbolsAndRefs(visitors.NodeVisitor):
            def visit_Node(self, node: concepts.Node, *, symbols: Set[str], refs: Set[str]) -> None:
                for name, metadata in node.__node_children__.items():
                    type_ = metadata["definition"].type_
                    if isinstance(type_, type):
                        if issubclass(type_, gtir.SymbolName):
                            symbols.add(getattr(node, name))
                        elif issubclass(type_, common.SymbolRef):
                            refs.add(getattr(node, name))
                self.generic_visit(node, symbols=symbols, refs=refs)

        symbols: Set[str] = set()
        refs: Set[str] = set()
        CollectSymbolsAndRefs().visit(self.vertical_loops, symbols=symbols, refs=refs)
        return [FieldDeclFactory(name=name) for name in refs - symbols]
