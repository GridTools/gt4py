# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List

import factory

from gt4py.cartesian.gtc import common, oir

from .common_utils import (
    CartesianOffsetFactory,
    HorizontalMaskFactory,
    identifier,
    undefined_symbol_list,
)


class FieldAccessFactory(factory.Factory):
    class Meta:
        model = oir.FieldAccess

    name = identifier(oir.FieldAccess)
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = common.DataType.FLOAT32


class ScalarAccessFactory(factory.Factory):
    class Meta:
        model = oir.ScalarAccess

    name = identifier(oir.ScalarAccess)
    dtype = common.DataType.FLOAT32


class BinaryOpFactory(factory.Factory):
    class Meta:
        model = oir.BinaryOp

    op = common.ArithmeticOperator.ADD
    left = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.FLOAT32)
    right = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.FLOAT32)


class VariableKOffsetFactory(factory.Factory):
    class Meta:
        model = oir.VariableKOffset

    k = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.INT32)


class LiteralFactory(factory.Factory):
    class Meta:
        model = oir.Literal

    value = "42"
    dtype = common.DataType.FLOAT32


class AssignStmtFactory(factory.Factory):
    class Meta:
        model = oir.AssignStmt

    left = factory.SubFactory(FieldAccessFactory)
    right = factory.SubFactory(FieldAccessFactory)


class MaskStmtFactory(factory.Factory):
    class Meta:
        model = oir.MaskStmt

    mask = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.BOOL)
    body = factory.List([factory.SubFactory(AssignStmtFactory)])


class HorizontalRestrictionFactory(factory.Factory):
    class Meta:
        model = oir.HorizontalRestriction

    mask = factory.SubFactory(HorizontalMaskFactory)
    body: List[oir.Stmt] = factory.List([factory.SubFactory(AssignStmtFactory)])


class WhileFactory(factory.Factory):
    class Meta:
        model = oir.While

    cond = factory.SubFactory(FieldAccessFactory, dtype=common.DataType.BOOL)
    body = factory.List([factory.SubFactory(AssignStmtFactory)])


class NativeFuncCallFactory(factory.Factory):
    class Meta:
        model = oir.NativeFuncCall

    func = common.NativeFunction.ABS
    args = factory.List([factory.SubFactory(FieldAccessFactory)])


class TemporaryFactory(factory.Factory):
    class Meta:
        model = oir.Temporary

    name = identifier(oir.Temporary)
    dtype = common.DataType.FLOAT32
    dimensions = (True, True, True)


class LocalScalarFactory(factory.Factory):
    class Meta:
        model = oir.LocalScalar

    name = identifier(oir.LocalScalar)
    dtype = common.DataType.FLOAT32


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = oir.FieldDecl

    name = identifier(oir.FieldDecl)
    dtype = common.DataType.FLOAT32
    dimensions = (True, True, True)


class HorizontalExecutionFactory(factory.Factory):
    class Meta:
        model = oir.HorizontalExecution

    body = factory.List([factory.SubFactory(AssignStmtFactory)])
    declarations: List[oir.LocalScalar] = []


class IntervalFactory(factory.Factory):
    class Meta:
        model = oir.Interval

    start = common.AxisBound.start()
    end = common.AxisBound.end()


class IJCacheFactory(factory.Factory):
    class Meta:
        model = oir.IJCache


class KCacheFactory(factory.Factory):
    class Meta:
        model = oir.KCache

    fill = True
    flush = True


class VerticalLoopSectionFactory(factory.Factory):
    class Meta:
        model = oir.VerticalLoopSection

    interval = factory.SubFactory(IntervalFactory)
    horizontal_executions = factory.List([factory.SubFactory(HorizontalExecutionFactory)])


class VerticalLoopFactory(factory.Factory):
    class Meta:
        model = oir.VerticalLoop

    loop_order = common.LoopOrder.PARALLEL
    sections = factory.List([factory.SubFactory(VerticalLoopSectionFactory)])
    caches: List[oir.CacheDesc] = []


class StencilFactory(factory.Factory):
    class Meta:
        model = oir.Stencil

    name = identifier(oir.Stencil)
    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])
    declarations: List[oir.Temporary] = []
    params = undefined_symbol_list(
        lambda name: FieldDeclFactory(name=name), "vertical_loops", "declarations"
    )
