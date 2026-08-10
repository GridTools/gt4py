# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union, cast

import factory

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.numpy import npir

from .common_utils import identifier, undefined_symbol_list


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = npir.FieldDecl

    name = identifier(npir.FieldDecl)
    dimensions = (True, True, True)
    data_dims: Tuple[int] = cast(Tuple[int], tuple())
    extent: Extent = Extent.zeros(ndims=2)
    dtype = common.DataType.FLOAT32


class TemporaryDeclFactory(factory.Factory):
    class Meta:
        model = npir.TemporaryDecl

    name = identifier(npir.TemporaryDecl)
    dtype = common.DataType.FLOAT32
    dimensions = (True, True, True)
    offset = (0, 0)
    padding = (0, 0)


class ScalarDeclFactory(factory.Factory):
    class Meta:
        model = npir.ScalarDecl

    name = identifier(npir.ScalarDecl)
    dtype = common.DataType.FLOAT32


class FieldSliceFactory(factory.Factory):
    class Meta:
        model = npir.FieldSlice

    name = identifier(npir.FieldSlice)
    i_offset: int = 0
    j_offset: int = 0
    k_offset: Union[int, npir.VarKOffset] = 0
    data_index: List[npir.Expr] = []
    dtype = common.DataType.FLOAT32


class ParamAccessFactory(factory.Factory):
    class Meta:
        model = npir.ParamAccess

    name = identifier(npir.ParamAccess)


class LocalScalarAccessFactory(factory.Factory):
    class Meta:
        model = npir.LocalScalarAccess

    name = identifier(npir.LocalScalarAccess)
    dtype = common.DataType.FLOAT32


class NativeFuncCallFactory(factory.Factory):
    class Meta:
        model = npir.NativeFuncCall

    func = common.NativeFunction.MIN
    args = factory.List([factory.SubFactory(FieldSliceFactory)])


class VectorAssignFactory(factory.Factory):
    class Meta:
        model = npir.VectorAssign

    left = factory.SubFactory(FieldSliceFactory)
    right = factory.SubFactory(FieldSliceFactory)
    horizontal_mask: Optional[common.HorizontalMask] = None


class VectorArithmeticFactory(factory.Factory):
    class Meta:
        model = npir.VectorArithmetic

    op = common.ArithmeticOperator.ADD
    left = factory.SubFactory(FieldSliceFactory)
    right = factory.SubFactory(FieldSliceFactory)


class HorizontalBlockFactory(factory.Factory):
    class Meta:
        model = npir.HorizontalBlock

    body = factory.List([factory.SubFactory(VectorAssignFactory)])
    extent: Extent = Extent.zeros(ndims=2)
    declarations: List[npir.LocalScalarDecl] = []


class VerticalPassFactory(factory.Factory):
    class Meta:
        model = npir.VerticalPass

    body = factory.List([factory.SubFactory(HorizontalBlockFactory)])
    lower = common.AxisBound.start()
    upper = common.AxisBound.end()
    direction = common.LoopOrder.PARALLEL


class ComputationFactory(factory.Factory):
    class Meta:
        model = npir.Computation

    arguments = factory.lazy_attribute(
        lambda node: [d.name for d in node.api_field_decls] + [d.name for d in node.param_decls]
    )
    param_decls: List[npir.ScalarDecl] = []
    temp_decls: List[npir.TemporaryDecl] = []
    vertical_passes = factory.List([factory.SubFactory(VerticalPassFactory)])
    api_field_decls = undefined_symbol_list(
        lambda name: FieldDeclFactory(name=name), "vertical_passes", "param_decls", "temp_decls"
    )
