# -*- coding: utf-8 -*-
#
# Copy stencil with temporary field

from gtc_unstructured.frontend.gtscript import FORWARD, Cell, Field, computation, location
from gtc_unstructured.irs.common import DataType


dtype = DataType.FLOAT64


def sten(field_in: Field[Cell, dtype], field_out: Field[Cell, dtype]):
    with computation(FORWARD), location(Cell):
        tmp = field_in
    with computation(FORWARD), location(Cell):
        field_out = tmp  # noqa: F841


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
