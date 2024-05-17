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

from gt4py.next.common import NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    Edge,
    IDim,
    MeshDescriptor,
    Vertex,
    simple_mesh,
)
from next_tests.integration_tests.cases import EField, IFloatField, VField

import numpy as np
import pytest

dace = pytest.importorskip("dace")


N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
CFTYPE = ts.FieldType(dims=[Cell], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
EFTYPE = ts.FieldType(dims=[Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
VFTYPE = ts.FieldType(dims=[Vertex], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
SIMPLE_MESH: MeshDescriptor = simple_mesh()
FSYMBOLS = dict(
    __w_size_0=N,
    __w_stride_0=1,
    __x_size_0=N,
    __x_stride_0=1,
    __y_size_0=N,
    __y_stride_0=1,
    __z_size_0=N,
    __z_stride_0=1,
    size=N,
)
CSYMBOLS = dict(
    ncells=SIMPLE_MESH.num_cells,
    nedges=SIMPLE_MESH.num_edges,
    nvertices=SIMPLE_MESH.num_vertices,
    __cells_size_0=SIMPLE_MESH.num_cells,
    __cells_stride_0=1,
    __edges_size_0=SIMPLE_MESH.num_edges,
    __edges_stride_0=1,
    __vertices_size_0=SIMPLE_MESH.num_vertices,
    __vertices_stride_0=1,
    __connectivity_C2E_size_0=SIMPLE_MESH.num_cells,
    __connectivity_C2E_size_1=SIMPLE_MESH.offset_provider["C2E"].max_neighbors,
    __connectivity_C2E_stride_0=SIMPLE_MESH.offset_provider["C2E"].max_neighbors,
    __connectivity_C2E_stride_1=1,
    __connectivity_C2V_size_0=SIMPLE_MESH.num_cells,
    __connectivity_C2V_size_1=SIMPLE_MESH.offset_provider["C2V"].max_neighbors,
    __connectivity_C2V_stride_0=SIMPLE_MESH.offset_provider["C2V"].max_neighbors,
    __connectivity_C2V_stride_1=1,
    __connectivity_E2V_size_0=SIMPLE_MESH.num_edges,
    __connectivity_E2V_size_1=SIMPLE_MESH.offset_provider["E2V"].max_neighbors,
    __connectivity_E2V_stride_0=SIMPLE_MESH.offset_provider["E2V"].max_neighbors,
    __connectivity_E2V_stride_1=1,
    __connectivity_V2E_size_0=SIMPLE_MESH.num_vertices,
    __connectivity_V2E_size_1=SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
    __connectivity_V2E_stride_0=SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
    __connectivity_V2E_stride_1=1,
)


def test_gtir_copy():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.deref("a")),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    arg_types = [IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, y=b, **FSYMBOLS)
    assert np.allclose(a, b)


def test_gtir_update():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.plus(im.deref("a"), 1.0)),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="x"),
            )
        ],
    )

    a = np.random.rand(N)
    ref = a + 1.0

    arg_types = [IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, **FSYMBOLS)
    assert np.allclose(a, ref)


def test_gtir_sum2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="z"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )("x", "y"),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    arg_types = [IFTYPE, IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, y=b, z=c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="z"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )("x", "x"),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    arg_types = [IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, z=b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
            domain,
        )
    )(
        "x",
        im.call(
            im.call("as_fieldop")(
                im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                domain,
            )
        )("y", "w"),
    )
    stencil2 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "b", "c")(
                im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c")))
            ),
            domain,
        )
    )("x", "y", "w")

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    arg_types = [IFTYPE, IFTYPE, IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]

    for i, stencil in enumerate([stencil1, stencil2]):
        testee = itir.Program(
            id=f"sum_3fields_{i}",
            function_definitions=[],
            params=[
                itir.Sym(id="x"),
                itir.Sym(id="y"),
                itir.Sym(id="w"),
                itir.Sym(id="z"),
                itir.Sym(id="size"),
            ],
            declarations=[],
            body=[
                itir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=itir.SymRef(id="z"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

        d = np.empty_like(a)

        sdfg(x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


def test_gtir_select():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="select_2sums",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="y"),
            itir.Sym(id="w"),
            itir.Sym(id="z"),
            itir.Sym(id="cond"),
            itir.Sym(id="scalar"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )(
                    "x",
                    im.call(
                        im.call("select")(
                            im.deref("cond"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                    domain,
                                )
                            )("y", "scalar"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                    domain,
                                )
                            )("w", "scalar"),
                        )
                    )(),
                ),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    arg_types = [
        IFTYPE,
        IFTYPE,
        IFTYPE,
        IFTYPE,
        ts.ScalarType(ts.ScalarKind.BOOL),
        ts.ScalarType(ts.ScalarKind.FLOAT64),
        ts.ScalarType(ts.ScalarKind.INT32),
    ]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    for s in [False, True]:
        d = np.empty_like(a)
        sdfg(cond=np.bool_(s), scalar=1.0, x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + 1) if s else (a + c + 1))


def test_gtir_select_nested():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="select_nested",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="z"),
            itir.Sym(id="cond_1"),
            itir.Sym(id="cond_2"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("select")(
                        im.deref("cond_1"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("a")(im.plus(im.deref("a"), 1)),
                                domain,
                            )
                        )("x"),
                        im.call(
                            im.call("select")(
                                im.deref("cond_2"),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a")(im.plus(im.deref("a"), 2)),
                                        domain,
                                    )
                                )("x"),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a")(im.plus(im.deref("a"), 3)),
                                        domain,
                                    )
                                )("x"),
                            )
                        )(),
                    )
                )(),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)

    arg_types = [
        IFTYPE,
        IFTYPE,
        ts.ScalarType(ts.ScalarKind.BOOL),
        ts.ScalarType(ts.ScalarKind.BOOL),
        ts.ScalarType(ts.ScalarKind.INT32),
    ]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    for s1 in [False, True]:
        for s2 in [False, True]:
            b = np.empty_like(a)
            sdfg(cond_1=np.bool_(s1), cond_2=np.bool_(s2), x=a, z=b, **FSYMBOLS)
            assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))


def test_gtir_cartesian_shift():
    DELTA = 3
    OFFSET = 1
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )

    # cartesian shift with literal integer offset
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a")(im.plus(im.deref(im.shift("IDim", OFFSET)("a")), DELTA)),
            domain,
        )
    )("x")

    # use dynamic offset retrieved from field
    stencil2 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "off")(
                im.plus(im.deref(im.shift("IDim", im.deref("off"))("a")), DELTA)
            ),
            domain,
        )
    )("x", "x_offset")

    # use the result of an arithmetic field operation as dynamic offset
    stencil3 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "off")(
                im.plus(im.deref(im.shift("IDim", im.plus(im.deref("off"), 0))("a")), DELTA)
            ),
            domain,
        )
    )("x", "x_offset")

    a = np.random.rand(N + OFFSET)
    a_offset = np.full(N, OFFSET, dtype=np.int32)
    b = np.empty(N)

    IOFFSET_FTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))

    arg_types = [IFTYPE, IOFFSET_FTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    offset_provider = {"IDim": IDim}

    for i, stencil in enumerate([stencil1, stencil2, stencil3]):
        testee = itir.Program(
            id=f"cartesian_shift_{i}",
            function_definitions=[],
            params=[
                itir.Sym(id="x"),
                itir.Sym(id="x_offset"),
                itir.Sym(id="y"),
                itir.Sym(id="size"),
            ],
            declarations=[],
            body=[
                itir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=itir.SymRef(id="y"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, offset_provider)

        FSYMBOLS_tmp = FSYMBOLS.copy()
        FSYMBOLS_tmp["__x_size_0"] = N + OFFSET
        FSYMBOLS_tmp["__x_offset_stride_0"] = 1
        sdfg(x=a, x_offset=a_offset, y=b, **FSYMBOLS_tmp)
        assert np.allclose(a[OFFSET:] + DELTA, b)


def test_gtir_connectivity_shift():
    C2V_neighbor_idx = 1
    C2E_neighbor_idx = 2
    cell_domain = im.call("unstructured_domain")(
        im.call("named_range")(itir.AxisLiteral(value=Cell.value), 0, "ncells"),
    )
    # apply shift 2 times along different dimensions
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.deref(im.shift("C2V", C2V_neighbor_idx)(im.shift("C2E", C2E_neighbor_idx)("it")))
            ),
            cell_domain,
        )
    )("ve_field")

    # multi-dimensional shift in one function call
    stencil2 = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.deref(
                    im.call(
                        im.call("shift")(
                            im.ensure_offset("C2V"),
                            im.ensure_offset(C2V_neighbor_idx),
                            im.ensure_offset("C2E"),
                            im.ensure_offset(C2E_neighbor_idx),
                        )
                    )("it")
                )
            ),
            cell_domain,
        )
    )("ve_field")

    # again multi-dimensional shift in one function call, but this time with dynamic offset values
    stencil3 = im.call(
        im.call("as_fieldop")(
            im.lambda_("it", "c2e_off", "c2v_off")(
                im.deref(
                    im.call(
                        im.call("shift")(
                            im.ensure_offset("C2V"),
                            im.deref("c2v_off"),
                            im.ensure_offset("C2E"),
                            im.plus(im.deref("c2e_off"), 0),
                        )
                    )("it")
                )
            ),
            cell_domain,
        )
    )("ve_field", "c2e_offset", "c2v_offset")

    ve = np.random.rand(SIMPLE_MESH.num_vertices, SIMPLE_MESH.num_edges)
    VE_FTYPE = ts.FieldType(dims=[Vertex, Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    CELL_OFFSET_FTYPE = ts.FieldType(dims=[Cell], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))

    arg_types = [
        VE_FTYPE,
        CELL_OFFSET_FTYPE,
        CELL_OFFSET_FTYPE,
        CFTYPE,
        ts.ScalarType(ts.ScalarKind.INT32),
    ]

    connectivity_C2E = SIMPLE_MESH.offset_provider["C2E"]
    assert isinstance(connectivity_C2E, NeighborTable)
    connectivity_C2V = SIMPLE_MESH.offset_provider["C2V"]
    assert isinstance(connectivity_C2V, NeighborTable)
    ref = ve[
        connectivity_C2V.table[:, C2V_neighbor_idx], connectivity_C2E.table[:, C2E_neighbor_idx]
    ]

    for i, stencil in enumerate([stencil1, stencil2, stencil3]):
        testee = itir.Program(
            id=f"connectivity_shift_2d_{i}",
            function_definitions=[],
            params=[
                itir.Sym(id="ve_field"),
                itir.Sym(id="c2e_offset"),
                itir.Sym(id="c2v_offset"),
                itir.Sym(id="cells"),
                itir.Sym(id="ncells"),
            ],
            declarations=[],
            body=[
                itir.SetAt(
                    expr=stencil,
                    domain=cell_domain,
                    target=itir.SymRef(id="cells"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, SIMPLE_MESH.offset_provider)

        c = np.empty(SIMPLE_MESH.num_cells)

        sdfg(
            ve_field=ve,
            c2e_offset=np.full(SIMPLE_MESH.num_cells, C2E_neighbor_idx, dtype=np.int32),
            c2v_offset=np.full(SIMPLE_MESH.num_cells, C2V_neighbor_idx, dtype=np.int32),
            cells=c,
            connectivity_C2E=connectivity_C2E.table,
            connectivity_C2V=connectivity_C2V.table,
            **FSYMBOLS,
            **CSYMBOLS,
            __ve_field_size_0=SIMPLE_MESH.num_vertices,
            __ve_field_size_1=SIMPLE_MESH.num_edges,
            __ve_field_stride_0=SIMPLE_MESH.num_edges,
            __ve_field_stride_1=1,
            __c2e_offset_stride_0=1,
            __c2v_offset_stride_0=1,
        )
        assert np.allclose(c, ref)


def test_gtir_connectivity_shift_chain():
    E2V_neighbor_idx = 1
    V2E_neighbor_idx = 2
    edge_domain = im.call("unstructured_domain")(
        im.call("named_range")(itir.AxisLiteral(value=Edge.value), 0, "nedges")
    )
    testee = itir.Program(
        id="connectivity_shift_chain",
        function_definitions=[],
        params=[
            itir.Sym(id="edges"),
            itir.Sym(id="edges_out"),
            itir.Sym(id="nedges"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(
                            im.deref(
                                im.shift("E2V", E2V_neighbor_idx)(
                                    im.shift("V2E", V2E_neighbor_idx)("it")
                                )
                            )
                        ),
                        edge_domain,
                    )
                )("edges"),
                domain=edge_domain,
                target=itir.SymRef(id="edges_out"),
            )
        ],
    )

    e = np.random.rand(SIMPLE_MESH.num_edges)
    e_out = np.empty_like(e)

    arg_types = [EFTYPE, EFTYPE, ts.ScalarType(ts.ScalarKind.INT32)]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, SIMPLE_MESH.offset_provider)

    assert isinstance(sdfg, dace.SDFG)
    connectivity_E2V = SIMPLE_MESH.offset_provider["E2V"]
    assert isinstance(connectivity_E2V, NeighborTable)
    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, NeighborTable)

    sdfg(
        edges=e,
        edges_out=e_out,
        connectivity_E2V=connectivity_E2V.table,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        **CSYMBOLS,
        __edges_out_size_0=CSYMBOLS["__edges_size_0"],
        __edges_out_stride_0=CSYMBOLS["__edges_stride_0"],
    )
    assert np.allclose(
        e_out,
        e[connectivity_V2E.table[connectivity_E2V.table[:, E2V_neighbor_idx], V2E_neighbor_idx]],
    )
