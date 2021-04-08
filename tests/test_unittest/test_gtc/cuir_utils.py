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

from gtc.cuir import cuir

from .common_utils import CartesianOffsetFactory, identifier, undefined_symbol_list


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = cuir.FieldDecl

    name = identifier(cuir.FieldDecl)
    dtype = cuir.DataType.FLOAT32


class TemporaryFactory(factory.Factory):
    class Meta:
        model = cuir.Temporary

    name = identifier(cuir.Temporary)
    dtype = cuir.DataType.FLOAT32


class IJExtentFactory(factory.Factory):
    class Meta:
        model = cuir.IJExtent

    i = (0, 0)
    j = (0, 0)


class IJCacheDeclFactory(factory.Factory):
    class Meta:
        model = cuir.IJCacheDecl

    name = identifier(cuir.IJCacheDecl)
    dtype = cuir.DataType.FLOAT32
    extent = factory.SubFactory(IJExtentFactory)


class FieldAccessFactory(factory.Factory):
    class Meta:
        model = cuir.FieldAccess

    name = identifier(cuir.FieldAccess)
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = cuir.DataType.FLOAT32


class IJCacheAccessFactory(factory.Factory):
    class Meta:
        model = cuir.IJCacheAccess

    name = identifier(cuir.IJCacheAccess)
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = cuir.DataType.FLOAT32


class AssignStmtFactory(factory.Factory):
    class Meta:
        model = cuir.AssignStmt

    left = factory.SubFactory(FieldAccessFactory)
    right = factory.SubFactory(FieldAccessFactory)


class HorizontalExecutionFactory(factory.Factory):
    class Meta:
        model = cuir.HorizontalExecution

    body = factory.List([factory.SubFactory(AssignStmtFactory)])
    mask = None
    declarations = factory.List([])
    extent = factory.SubFactory(IJExtentFactory)


class VerticalLoopSectionFactory(factory.Factory):
    class Meta:
        model = cuir.VerticalLoopSection

    start = cuir.AxisBound.start()
    end = cuir.AxisBound.end()
    horizontal_executions = factory.List([factory.SubFactory(HorizontalExecutionFactory)])


class VerticalLoopFactory(factory.Factory):
    class Meta:
        model = cuir.VerticalLoop

    loop_order = cuir.LoopOrder.PARALLEL
    sections = factory.List([factory.SubFactory(VerticalLoopSectionFactory)])
    ij_caches = factory.List([])
    k_caches = factory.List([])


class KernelFactory(factory.Factory):
    class Meta:
        model = cuir.Kernel

    vertical_loops = factory.List([factory.SubFactory(VerticalLoopFactory)])


class ProgramFactory(factory.Factory):
    class Meta:
        model = cuir.Program

    name = identifier(cuir.Program)
    params = undefined_symbol_list(
        lambda name: FieldDeclFactory(name=name), "kernels", "temporaries"
    )
    temporaries = factory.List([])
    kernels = factory.List([factory.SubFactory(KernelFactory)])
