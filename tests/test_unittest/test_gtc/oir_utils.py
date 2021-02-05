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

from typing import List

import factory

from gtc import common, oir

from .common_utils import CartesianOffsetFactory, undefined_symbol_list


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
    params = undefined_symbol_list(lambda name: FieldDeclFactory(name=name), "vertical_loops")
