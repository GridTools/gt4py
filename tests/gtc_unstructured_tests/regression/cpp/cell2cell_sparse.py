# GridTools Compiler Toolchain (GTC) - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GTC project and the GridTools framework.
# GTC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
# Cell to cell reduction.
# Note that the reduction refers to a LocationRef from outside!
#
# ```python
# for c1 in cells(mesh):
#     field1 = sum(f[c1,c2] for c2 in cells(c1))
# ```

import os

from devtools import debug

from gtc_unstructured.irs import gtir
from gtc_unstructured.irs.common import DataType, LocationType, LoopOrder
from gtc_unstructured.irs.gtir import (
    AssignStmt,
    Dimensions,
    Domain,
    FieldAccess,
    HorizontalDimension,
    HorizontalLoop,
    LocationComprehension,
    LocationRef,
    NeighborChain,
    NeighborReduce,
    ReduceOperator,
    Stencil,
    UField,
    VerticalLoop,
)
from gtc_unstructured.irs.gtir_to_nir import GtirToNir
from gtc_unstructured.irs.nir_to_usid import NirToUsid
from gtc_unstructured.irs.usid_codegen import UsidGpuCodeGenerator


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


sten = Stencil(
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

comp = gtir.Computation(name="sten", params=[field_in, field_out], stencils=[sten])
# debug(comp)

nir_comp = GtirToNir().visit(comp)
debug(nir_comp)
usid_comp = NirToUsid().visit(nir_comp)
debug(usid_comp)

generated_code = UsidGpuCodeGenerator.apply(usid_comp)
print(generated_code)

output_file = os.path.dirname(os.path.realpath(__file__)) + "/generated_cell2cell_sparse.hpp"
with open(output_file, "w+") as output:
    output.write(generated_code)
