# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List

import factory

from gt4py.cartesian.gtc.cuir import cuir

from .common_utils import CartesianOffsetFactory, identifier, undefined_symbol_list


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = cuir.FieldDecl

    name = identifier(cuir.FieldDecl)
    dtype = cuir.DataType.FLOAT32
    dimensions = (True, True, True)


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
    positionals: List[cuir.Positional] = []
    temporaries = factory.List([])
    kernels = factory.List([factory.SubFactory(KernelFactory)])
    axis_sizes = cuir.axis_size_decls()
