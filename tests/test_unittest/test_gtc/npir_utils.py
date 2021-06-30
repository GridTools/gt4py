# -*- coding: utf-8 -*-
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
