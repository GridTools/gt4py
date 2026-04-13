# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List

import factory

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.gtcpp import gtcpp

from .common_utils import CartesianOffsetFactory, identifier, undefined_symbol_list


class AccessorRefFactory(factory.Factory):
    class Meta:
        model = gtcpp.AccessorRef

    name = identifier(gtcpp.AccessorRef)
    offset = factory.SubFactory(CartesianOffsetFactory)
    dtype = common.DataType.FLOAT32


class AssignStmtFactory(factory.Factory):
    class Meta:
        model = gtcpp.AssignStmt

    left = factory.SubFactory(AccessorRefFactory)
    right = factory.SubFactory(AccessorRefFactory)


class GTLevelFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTLevel

    splitter = 0
    offset = 0


class GTIntervalFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTInterval

    from_level = factory.SubFactory(GTLevelFactory, splitter=0, offset=1)
    to_level = factory.SubFactory(GTLevelFactory, splitter=1, offset=-1)


class GTApplyMethodFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTApplyMethod

    interval = factory.SubFactory(GTIntervalFactory)
    body = factory.List([factory.SubFactory(AssignStmtFactory)])
    local_variables: List[gtcpp.LocalVarDecl] = []


class LiteralFactory(factory.Factory):
    class Meta:
        model = gtcpp.Literal

    value = "42.0"
    dtype = common.DataType.FLOAT32


class BlockStmtFactory(factory.Factory):
    class Meta:
        model = gtcpp.BlockStmt

    body: List[gtcpp.Stmt] = []


class IfStmtFactory(factory.Factory):
    class Meta:
        model = gtcpp.IfStmt

    cond = factory.SubFactory(LiteralFactory, value="true", dtype=common.DataType.BOOL)
    true_branch = factory.SubFactory(BlockStmtFactory)
    false_branch = None


class GTExtentFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTExtent

    i = (0, 0)
    j = (0, 0)
    k = (0, 0)


class GTAccessorFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTAccessor

    name = identifier(gtcpp.GTAccessor)
    id = factory.Sequence(lambda i: i)  # noqa: A003
    intent = gtcpp.Intent.INOUT
    extent = factory.SubFactory(GTExtentFactory)
    ndim: int = 3


class GTParamListFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTParamList

    accessors: List[gtcpp.GTAccessor] = []


class GTFunctorFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTFunctor

    name = identifier(gtcpp.GTFunctor)
    applies = factory.List([factory.SubFactory(GTApplyMethodFactory)])
    param_list = undefined_symbol_list(
        lambda name: GTAccessorFactory(name=name),
        "applies",
        list_creator=lambda lst: GTParamListFactory(accessors=lst),
    )


class ArgFactory(factory.Factory):
    class Meta:
        model = gtcpp.Arg

    name = identifier(gtcpp.Arg)


class GTStageFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTStage

    functor = identifier(gtcpp.GTStage)
    args: List[gtcpp.Arg] = factory.List([factory.SubFactory(ArgFactory)])


class GTMultiStageFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTMultiStage

    loop_order = common.LoopOrder.PARALLEL
    stages = factory.List([factory.SubFactory(GTStageFactory)])
    caches: List[gtcpp.IJCache] = []


class GTComputationCallFactory(factory.Factory):
    class Meta:
        model = gtcpp.GTComputationCall

    arguments: List[gtcpp.Arg] = []
    extra_decls: List[gtcpp.ComputationDecl] = []
    multi_stages = factory.List([factory.SubFactory(GTMultiStageFactory)])
    temporaries = undefined_symbol_list(lambda name: TemporaryFactory(name=name), "multi_stages")


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = gtcpp.FieldDecl

    name = identifier(gtcpp.FieldDecl)
    dtype = common.DataType.FLOAT32
    dimensions = (True, True, True)


class TemporaryFactory(factory.Factory):
    class Meta:
        model = gtcpp.Temporary

    name = identifier(gtcpp.Temporary)
    dimensions = (True, True, True)
    dtype = common.DataType.FLOAT32


class ProgramFactory(factory.Factory):
    class Meta:
        model = gtcpp.Program

    name = identifier(gtcpp.Program)
    functors = factory.List([factory.SubFactory(GTFunctorFactory)])
    gt_computation = factory.SubFactory(GTComputationCallFactory)
    parameters = undefined_symbol_list(
        lambda name: FieldDeclFactory(name=name), "functors", "gt_computation"
    )
