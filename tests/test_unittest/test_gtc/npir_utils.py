from gt4py.gtc.python import npir


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

