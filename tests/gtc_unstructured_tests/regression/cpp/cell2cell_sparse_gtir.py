# -*- coding: utf-8 -*-
#
# Cell to cell reduction.
# Note that the reduction refers to a LocationRef from outside!
#
# ```python
# for c1 in cells(mesh):
#     field1 = sum(f[c1,c2] for c2 in cells(c1))
# ```

# TODO fix

from gtc_unstructured.irs import gtir
from gtc_unstructured.irs.common import DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    Dimensions,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationComprehension,
    LocationRef,
    NeighborReduce,
    ReduceOperator,
    Stencil,
    UField,
    VerticalLoop,
)

field_in = UField(
    name="field_in",
    dimensions=Dimensions(
        horizontal=HorizontalDimension(
            primary=LocationType.Cell,
            secondary=NeighborChain(elements=[LocationType.Cell, LocationType.Cell]),
        )
    ),
    vtype=DataType.FLOAT64,
)
field_out = UField(
    name="field_out",
    dimensions=Dimensions(horizontal=HorizontalDimension(primary=LocationType.Cell)),
    vtype=DataType.FLOAT64,
)

red_c2c = NeighborReduce(
    op=ReduceOperator.ADD,
    operand=FieldAccess(
        name="field_in",
        location_type=LocationType.Cell,
        subscript=[LocationRef(name="c1"), LocationRef(name="c2")],
    ),
    neighbors=LocationComprehension(
        name="c2",
        chain=NeighborChain(elements=[LocationType.Cell, LocationType.Cell]),
        of=LocationRef(name="c1"),
    ),
    location_type=LocationType.Cell,
)

assign_c2c_red = AssignStmt(
    left=FieldAccess(
        name="field_out", subscript=[LocationRef(name="c1")], location_type=LocationType.Cell
    ),
    right=red_c2c,
)


stencil = Stencil(
    vertical_loops=[
        VerticalLoop(
            loop_order=LoopOrder.FORWARD,
            horizontal_loops=[
                HorizontalLoop(
                    location=LocationComprehension(
                        name="c1", chain=NeighborChain(elements=[LocationType.Cell]), of=Domain()
                    ),
                    stmt=assign_c2c_red,
                )
            ],
        )
    ],
)

sten = gtir.Computation(name="sten", params=[field_in, field_out], stencils=[stencil])

if __name__ == "__main__":
    import generator

    generator.default_main(sten)
