# -*- coding: utf-8 -*-
from typing import Tuple

import factory

from gtc.python import npir


class FieldSliceBuilder:
    def __init__(self, name: str, *, parallel_k=False):
        self._name = name
        self._offsets = [0, 0, 0]
        self._parallel_k = parallel_k

    def offsets(self, i: int, j: int, k: int):
        self._offsets = [i, j, k]
        return self

    def build(self):
        return npir.FieldSlice(
            name=self._name,
            i_offset=npir.AxisOffset.i(self._offsets[0]),
            j_offset=npir.AxisOffset.j(self._offsets[1]),
            k_offset=npir.AxisOffset.k(self._offsets[2], parallel=self._parallel_k),
        )


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
