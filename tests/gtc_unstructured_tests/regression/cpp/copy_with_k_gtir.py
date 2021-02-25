# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

from devtools import debug

from gtc_unstructured.irs.common import DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    Computation,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationRef,
    PrimaryLocation,
    Stencil,
    UField,
    VerticalDimension,
    VerticalLoop,
)


dtype = DataType.FLOAT64


params = [
    UField(
        name="in_field",
        vtype=dtype,
        dimensions=Dimensions(
            horizontal=HorizontalDimension(primary=LocationType.Vertex),
            vertical=VerticalDimension(),
        ),
    ),
    UField(
        name="out_field",
        vtype=dtype,
        dimensions=Dimensions(
            horizontal=HorizontalDimension(primary=LocationType.Edge),
            vertical=VerticalDimension(),
        ),
    ),
]

hloop = HorizontalLoop(
    stmt=AssignStmt(
        left=FieldAccess(
            name="out_field",
            subscript=[LocationRef(name="edge")],
            location_type=LocationType.Edge,
        ),
        right=FieldAccess(
            name="in_field",
            subscript=[LocationRef(name="edge")],
            location_type=LocationType.Edge,
        ),
    ),
    location=PrimaryLocation(name="edge", location_type=LocationType.Edge),
)

stencils = [
    Stencil(vertical_loops=[VerticalLoop(horizontal_loops=[hloop], loop_order=LoopOrder.FORWARD)])
]

sten = Computation(name="sten", connectivities=[], params=params, stencils=stencils)


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
