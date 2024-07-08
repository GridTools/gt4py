# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
"""
Test that ITIR can be lowered to SDFG.

Note: this test module covers the fieldview flavour of ITIR.
"""

import copy
from gt4py.next.common import NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from functools import reduce
import numpy as np

from simple_icon_mesh import (
    IDim,  # Dimensions
    JDim,
    KDim,
    EdgeDim,
    VertexDim,
    CellDim,
    ECVDim,
    E2C2VDim,
    NbCells,  # Constants of the size
    NbEdges,
    NbVertices,
    E2C2VDim,  # Offsets
    E2C2V,
    SIZE_TYPE,  # Type definitions
    E2C2V_connectivity,
    E2ECV_connectivity,
    make_syms,  # Helpers
)

# For cartesian stuff.
N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
IJFTYPE = ts.FieldType(dims=[IDim, JDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))


######################
#   TESTS


def gtir_copy3():
    # We can not use the size symbols inside the domain
    #  Because the translator complains.

    # Input domain
    input_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value, kind=IDim.kind), 0, "org_sizeI"),
        im.call("named_range")(itir.AxisLiteral(value=JDim.value, kind=JDim.kind), 0, "org_sizeJ"),
    )

    # Domain for after we have processed the IDim.
    first_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value, kind=IDim.kind), 0, "sizeI"),
        im.call("named_range")(itir.AxisLiteral(value=JDim.value, kind=JDim.kind), 0, "org_sizeJ"),
    )

    # This is the final domain, or after we have removed the JDim
    final_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value, kind=IDim.kind), 0, "sizeI"),
        im.call("named_range")(itir.AxisLiteral(value=JDim.value, kind=JDim.kind), 0, "sizeJ"),
    )

    IOffset = 1
    JOffset = 2

    testee = itir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IJFTYPE),
            itir.Sym(id="y", type=IJFTYPE),
            itir.Sym(id="sizeI", type=SIZE_TYPE),
            itir.Sym(id="sizeJ", type=SIZE_TYPE),
            itir.Sym(id="org_sizeI", type=SIZE_TYPE),
            itir.Sym(id="org_sizeJ", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    # This processed the `JDim`, it is first because its arguments are
                    #  evaluated before, so there the first cutting is happening.
                    im.call("as_fieldop")(
                        im.lambda_("a")(
                            im.deref(
                                im.shift("JDim", JOffset)("a")  # This does not work
                            )
                        ),
                        final_domain,
                    )
                )(
                    # Now here we will process the `IDim` part.
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("b")(
                                im.deref(
                                    im.shift("IDim", IOffset)("b")
                                    # "b"
                                )
                            ),
                            first_domain,
                        )
                    )("x"),
                ),
                domain=final_domain,
                target=itir.SymRef(id="y"),
            )
        ],
    )

    # We only need an offset provider for the translation.
    offset_provider = {
        "IDim": IDim,
        "JDim": JDim,
    }

    sdfg = dace_backend.build_sdfg_from_gtir(
        testee,
        offset_provider,
    )

    output_size_I, output_size_J = 10, 10
    input_size_I, input_size_J = 20, 20

    a = np.random.rand(input_size_I, input_size_J)
    b = np.empty((output_size_I, output_size_J), dtype=np.float64)

    SYMBS = make_syms(x=a, y=b)

    sdfg(
        x=a,
        y=b,
        sizeI=output_size_I,
        sizeJ=output_size_J,
        org_sizeI=input_size_I,
        org_sizeJ=input_size_J,
        **SYMBS,
    )

    ref = a[IOffset : (IOffset + output_size_I), JOffset : (JOffset + output_size_J)]

    assert np.all(b == ref)
    assert True


def gtir_ecv_shift():
    # EdgeDim, E2C2VDim
    domain = im.call("unstructured_domain")(
        im.call("named_range")(
            itir.AxisLiteral(value=EdgeDim.value, kind=EdgeDim.kind), 0, "nedges"
        ),
        # im.call("named_range")(itir.AxisLiteral(value=E2C2VDim.value, kind=E2C2VDim.kind), 0, 4),
    )

    INPUT_FTYPE = ts.FieldType(dims=[ECVDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    OUTPUT_FTYPE = ts.FieldType(dims=[EdgeDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))

    testee = itir.Program(
        id="gtir_shift",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=INPUT_FTYPE),
            itir.Sym(id="y", type=OUTPUT_FTYPE),
            itir.Sym(id="nedges", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    # This processed the `JDim`, it is first because its arguments are
                    #  evaluated before, so there the first cutting is happening.
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.deref(im.shift("E2ECV", 0)("a"))),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="y"),
            )
        ],
    )

    offset_provider = {
        "E2C2V": E2C2V_connectivity,
        "E2ECV": E2ECV_connectivity,
    }

    sdfg = dace_backend.build_sdfg_from_gtir(
        testee,
        offset_provider,
    )

    a = np.random.rand(NbEdges * 4)
    b = np.empty((NbEdges,), dtype=np.float64)

    call_args = {
        "x": a,
        "y": b,
        "connectivity_E2C2V": E2C2V_connectivity.table.copy(),
        "connectivity_E2ECV": E2ECV_connectivity.table.copy(),
    }

    SYMBS = make_syms(**call_args)

    sdfg(
        **call_args,
        nedges=NbEdges,
        **SYMBS,
    )
    ref = a[E2ECV_connectivity.table[:, 0]]

    assert np.allclose(ref, b)
    assert True


if "__main__" == __name__:
    # gtir_copy3()
    gtir_ecv_shift()
