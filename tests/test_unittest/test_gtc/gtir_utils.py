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

import factory

from gtc import common, gtir

from .common_utils import CartesianOffsetFactory, identifier, undefined_symbol_list


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
    dtype = common.DataType.FLOAT32


class ScalarAccessFactory(factory.Factory):
    class Meta:
        model = gtir.ScalarAccess

    name = identifier(gtir.ScalarAccess)
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

    name = identifier(gtir.FieldDecl)
    dtype = common.DataType.FLOAT32


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
    temporaries = []
    body = factory.List([factory.SubFactory(ParAssignStmtFactory)])


class StencilFactory(factory.Factory):
    class Meta:
        model = gtir.Stencil

    name = identifier(gtir.Stencil)
    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])
    params = undefined_symbol_list(lambda name: FieldDeclFactory(name=name), "vertical_loops")
