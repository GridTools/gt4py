# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, List

import factory

from gt4py.cartesian.gtc import common, gtir

from .common_utils import (
    CartesianOffsetFactory,
    HorizontalMaskFactory,
    identifier,
    undefined_symbol_list,
)


class LiteralFactory(factory.Factory):
    class Meta:
        model = gtir.Literal

    value = "42.0"
    dtype = common.DataType.FLOAT32


class FieldAccessFactory(factory.Factory):
    class Meta:
        model = gtir.FieldAccess

    name = identifier(gtir.FieldAccess)
    offset = factory.SubFactory(CartesianOffsetFactory)
    data_index: List[gtir.Expr] = []
    dtype = common.DataType.FLOAT32


class ScalarAccessFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarAccess

    name = identifier(gtir.ScalarAccess)
    dtype = common.DataType.FLOAT32


class VariableKOffsetFactory(factory.Factory):
    class Meta:
        model = gtir.VariableKOffset

    k = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.INT32)


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

    body: List[gtir.Stmt] = factory.List([factory.SubFactory(ParAssignStmtFactory)])


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


class HorizontalRestrictionFactory(factory.Factory):
    class Meta:
        model = gtir.HorizontalRestriction

    mask = factory.SubFactory(HorizontalMaskFactory)
    body: List[gtir.Stmt] = factory.List([factory.SubFactory(ParAssignStmtFactory)])


class WhileFactory(factory.Factory):
    class Meta:
        model = gtir.While

    cond = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.BOOL)
    body = factory.List([factory.SubFactory(ParAssignStmtFactory)])


class IntervalFactory(factory.Factory):
    class Meta:
        model = gtir.Interval

    start = common.AxisBound.start()
    end = common.AxisBound.end()


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = gtir.FieldDecl

    name = identifier(gtir.FieldDecl)
    dtype = common.DataType.FLOAT32
    dimensions = (True, True, True)


class ScalarDeclFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarDecl

    name = identifier(gtir.ScalarDecl)
    dtype = common.DataType.FLOAT32


class VerticalLoopFactory(factory.Factory):
    class Meta:
        model = gtir.VerticalLoop

    interval = factory.SubFactory(IntervalFactory)
    loop_order = common.LoopOrder.PARALLEL
    temporaries: List[gtir.FieldDecl] = []
    body = factory.List([factory.SubFactory(ParAssignStmtFactory)])


class StencilFactory(factory.Factory):
    class Meta:
        model = gtir.Stencil

    name = identifier(gtir.Stencil)
    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])
    params = undefined_symbol_list(lambda name: FieldDeclFactory(name=name), "vertical_loops")
    api_signature = factory.LazyAttribute(
        lambda x: [gtir.Argument(name=p.name, is_keyword=False, default="") for p in x.params]
    )
    externals: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    docstring: str = ""
