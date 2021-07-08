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

from typing import Tuple

import factory

from gtc import common
from gtc.python import npir


class FieldSliceFactory(factory.Factory):
    class Meta:
        model = npir.FieldSlice

    class Params:
        offsets: Tuple[int, int, int] = (0, 0, 0)
        parallel_k: bool = False

    name = factory.Sequence(lambda n: "field_%d" % n)
    i_offset = factory.LazyAttribute(lambda obj: npir.AxisOffset.i(obj.offsets[0]))
    j_offset = factory.LazyAttribute(lambda obj: npir.AxisOffset.j(obj.offsets[1]))
    k_offset = factory.LazyAttribute(
        lambda obj: npir.AxisOffset.k(obj.offsets[2], parallel=obj.parallel_k)
    )


class FieldDeclFactory(factory.Factory):
    class Meta:
        model = npir.FieldDecl

    name = factory.Sequence(lambda n: "field_%d" % n)
    dimensions = (True, True, True)
    dtype = common.DataType.FLOAT32


class NativeFuncCallFactory(factory.Factory):
    class Meta:
        model = npir.NativeFuncCall

    func = common.NativeFunction.MIN
    args = factory.List([factory.SubFactory(FieldSliceFactory)])


class VectorAssignFactory(factory.Factory):
    class Meta:
        model = npir.VectorAssign

    class Params:
        temp_name = factory.Sequence(lambda n: "field_%d" % n)
        temp_dtype = common.DataType.INT64
        temp_init = factory.Trait(
            left=factory.LazyAttribute(lambda obj: npir.VectorTemp(name=obj.temp_name)),
            right=factory.LazyAttribute(lambda obj: npir.EmptyTemp(dtype=obj.temp_dtype)),
        )

    left = factory.SubFactory(FieldSliceFactory)
    right = factory.SubFactory(FieldSliceFactory)


class VerticalPassFactory(factory.Factory):
    class Meta:
        model = npir.VerticalPass

    temp_defs = factory.List([factory.SubFactory(VectorAssignFactory, temp_init=True)])
    body = factory.List([factory.SubFactory(VectorAssignFactory)])
    lower = common.AxisBound.start()
    upper = common.AxisBound.end()
    direction = common.LoopOrder.PARALLEL
