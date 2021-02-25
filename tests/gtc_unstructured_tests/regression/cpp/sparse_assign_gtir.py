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
    Computation,
    Connectivity,
    ConnectivityRef,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationComprehension,
    LocationRef,
    NeighborAssignStmt,
    PrimaryLocation,
    SparseField,
    Stencil,
    UField,
    VerticalLoop,
)
from gtc_unstructured.irs.nir_passes.merge_horizontal_loops import find_and_merge_horizontal_loops


dtype = DataType.FLOAT64


conn = Connectivity(
    name="e2v",
    primary=LocationType.Edge,
    secondary=LocationType.Vertex,
    max_neighbors=2,
    has_skip_values=False,
)

params = [
    UField(
        name="in_field",
        vtype=dtype,
        dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Vertex)),
    ),
    SparseField(
        name="out_field",
        connectivity="e2v",
        vtype=dtype,
        dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Edge)),
    ),
]

hloop = HorizontalLoop(
    stmt=NeighborAssignStmt(
        left=FieldAccess(
            name="out_field",
            subscript=[LocationRef(name="edge")],
            location_type=LocationType.Edge,
        ),
        right=FieldAccess(
            name="in_field",
            subscript=[LocationRef(name="v_neighs_of_e")],
            location_type=LocationType.Edge,
        ),
        neighbors=LocationComprehension(name="v_neighs_of_e", of=ConnectivityRef(name="e2v")),
        location_type=LocationType.Edge,
    ),
    location=PrimaryLocation(name="edge", location_type=LocationType.Edge),
)

stencils = [
    Stencil(vertical_loops=[VerticalLoop(horizontal_loops=[hloop], loop_order=LoopOrder.FORWARD)])
]

sten = Computation(name="sten", connectivities=[conn], params=params, stencils=stencils)


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
