# -*- coding: utf-8 -*-
#
# Cell to cell reduction.
# Note that the reduction refers to a LocationRef from outside!
#
# ```python
# for c1 in cells(mesh):
#     field1 = sum(f[c1] * f[c2] for c2 in cells(c1))
# ```

import types

from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Cell,
    Connectivity,
    Field,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType


C2C = types.new_class("C2C", (Connectivity[Cell, Cell, 4, False],))

dtype = DataType.FLOAT64


def sten(c2c: C2C, field_in: Field[Cell, dtype], field_out: Field[Cell, dtype]):
    with computation(FORWARD), location(Cell) as c1:
        field_out[c1] = sum(field_in[c1] + field_in[c2] for c2 in c2c[c1])


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
