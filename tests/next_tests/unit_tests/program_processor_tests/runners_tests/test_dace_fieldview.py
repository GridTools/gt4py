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
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    Edge,
    IDim,
    MeshDescriptor,
    V2EDim,
    Vertex,
    simple_mesh,
    skip_value_mesh,
)
from functools import reduce
import numpy as np
import pytest

dace = pytest.importorskip("dace")


N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
CFTYPE = ts.FieldType(dims=[Cell], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
EFTYPE = ts.FieldType(dims=[Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
VFTYPE = ts.FieldType(dims=[Vertex], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
V2E_FTYPE = ts.FieldType(dims=[Vertex, V2EDim], dtype=EFTYPE.dtype)
SIMPLE_MESH: MeshDescriptor = simple_mesh()
SKIP_VALUE_MESH: MeshDescriptor = skip_value_mesh()
SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)
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


def make_mesh_symbols(mesh: MeshDescriptor):
    C2E_size_0, C2E_size_1 = mesh.offset_provider["C2E"].table.shape
    C2E_stride_0, C2E_stride_1 = C2E_size_1, 1  # mesh.offset_provider["C2E"].table.strides

    C2V_size_0, C2V_size_1 = mesh.offset_provider["C2V"].table.shape
    C2V_stride_0, C2V_stride_1 = C2V_size_1, 1  # mesh.offset_provider["C2V"].table.strides

    E2V_size_0, E2V_size_1 = mesh.offset_provider["E2V"].table.shape
    E2V_stride_0, E2V_stride_1 = E2V_size_1, 1  # mesh.offset_provider["E2V"].table.strides

    V2E_size_0, V2E_size_1 = mesh.offset_provider["V2E"].table.shape
    V2E_stride_0, V2E_stride_1 = V2E_size_1, 1  # mesh.offset_provider["V2E"].table.strides

    return dict(
        ncells=mesh.num_cells,
        nedges=mesh.num_edges,
        nvertices=mesh.num_vertices,
        __cells_size_0=mesh.num_cells,
        __cells_stride_0=1,
        __edges_size_0=mesh.num_edges,
        __edges_stride_0=1,
        __vertices_size_0=mesh.num_vertices,
        __vertices_stride_0=1,
        __connectivity_C2E_size_0=C2E_size_0,
        __connectivity_C2E_size_1=C2E_size_1,
        __connectivity_C2E_stride_0=C2E_stride_0,
        __connectivity_C2E_stride_1=C2E_stride_1,
        __connectivity_C2V_size_0=C2V_size_0,
        __connectivity_C2V_size_1=C2V_size_1,
        __connectivity_C2V_stride_0=C2V_stride_0,
        __connectivity_C2V_stride_1=C2V_stride_1,
        __connectivity_E2V_size_0=E2V_size_0,
        __connectivity_E2V_size_1=E2V_size_1,
        __connectivity_E2V_stride_0=E2V_stride_0,
        __connectivity_E2V_stride_1=E2V_stride_1,
        __connectivity_V2E_size_0=V2E_size_0,
        __connectivity_V2E_size_1=V2E_size_1,
        __connectivity_V2E_stride_0=V2E_stride_0,
        __connectivity_V2E_stride_1=V2E_stride_1,
    )


def test_gtir_copy():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = gtir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.deref("a")),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    sdfg(x=a, y=b, **FSYMBOLS)
    assert np.allclose(a, b)


def test_gtir_update():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    stencil1 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a")(im.plus(im.deref("a"), 1.0)),
            domain,
        )
    )("x")
    stencil2 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
            domain,
        )
    )("x", 1.0)

    for i, stencil in enumerate([stencil1, stencil2]):
        testee = gtir.Program(
            id=f"gtir_update_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=IFTYPE),
                gtir.Sym(id="size", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=gtir.SymRef(id="x"),
                )
            ],
        )
        sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

        a = np.random.rand(N)
        ref = a + 1.0

        sdfg(x=a, **FSYMBOLS)
        assert np.allclose(a, ref)


def test_gtir_sum2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = gtir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )("x", "y"),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    sdfg(x=a, y=b, z=c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = gtir.Program(
        id="sum_2fields_sym",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )("x", "x"),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    sdfg(x=a, z=b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
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

    for i, stencil in enumerate([stencil1, stencil2]):
        testee = gtir.Program(
            id=f"sum_3fields_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=IFTYPE),
                gtir.Sym(id="y", type=IFTYPE),
                gtir.Sym(id="w", type=IFTYPE),
                gtir.Sym(id="z", type=IFTYPE),
                gtir.Sym(id="size", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=gtir.SymRef(id="z"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

        d = np.empty_like(a)

        sdfg(x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


def test_gtir_cond():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = gtir.Program(
        id="cond_2sums",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="w", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="scalar", type=ts.ScalarType(ts.ScalarKind.FLOAT64)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )(
                    "x",
                    im.call(
                        im.call("cond")(
                            im.deref("pred"),
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
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    for s in [False, True]:
        d = np.empty_like(a)
        sdfg(pred=np.bool_(s), scalar=1.0, x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + 1) if s else (a + c + 1))


def test_gtir_cond_nested():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = gtir.Program(
        id="cond_nested",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="pred_1", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="pred_2", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("cond")(
                        im.deref("pred_1"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                domain,
                            )
                        )("x", 1),
                        im.call(
                            im.call("cond")(
                                im.deref("pred_2"),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                        domain,
                                    )
                                )("x", 2),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                        domain,
                                    )
                                )("x", 3),
                            )
                        )(),
                    )
                )(),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    for s1 in [False, True]:
        for s2 in [False, True]:
            b = np.empty_like(a)
            sdfg(pred_1=np.bool_(s1), pred_2=np.bool_(s2), x=a, z=b, **FSYMBOLS)
            assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))


def test_gtir_neighbors():
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
    )
    v2e_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
        im.call("named_range")(
            gtir.AxisLiteral(value=V2EDim.value, kind=V2EDim.kind),
            0,
            SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
        ),
    )
    testee = gtir.Program(
        id=f"neighbors",
        function_definitions=[],
        params=[
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="v2e_field", type=V2E_FTYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(im.neighbors("V2E", "it")),
                        vertex_domain,
                    )
                )("edges"),
                domain=v2e_domain,
                target=gtir.SymRef(id="v2e_field"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH.offset_provider)

    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v2e_field = np.empty([SIMPLE_MESH.num_vertices, connectivity_V2E.max_neighbors], dtype=e.dtype)

    sdfg(
        edges=e,
        v2e_field=v2e_field,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
        __v2e_field_size_0=SIMPLE_MESH.num_vertices,
        __v2e_field_size_1=connectivity_V2E.max_neighbors,
        __v2e_field_stride_0=connectivity_V2E.max_neighbors,
        __v2e_field_stride_1=1,
    )
    assert np.allclose(v2e_field, e[connectivity_V2E.table])


def test_gtir_reduce():
    init_value = np.random.rand()
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
    )
    stencil_inlined = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                    im.neighbors("V2E", "it")
                )
            ),
            vertex_domain,
        )
    )("edges")
    stencil_fieldview = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                    im.deref("it")
                )
            ),
            vertex_domain,
        )
    )(
        im.call(
            im.call("as_fieldop")(
                im.lambda_("it")(im.neighbors("V2E", "it")),
                vertex_domain,
            )
        )("edges")
    )

    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v_ref = [
        reduce(lambda x, y: x + y, e[v2e_neighbors], init_value)
        for v2e_neighbors in connectivity_V2E.table
    ]

    for i, stencil in enumerate([stencil_inlined, stencil_fieldview]):
        testee = gtir.Program(
            id=f"reduce_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="edges", type=EFTYPE),
                gtir.Sym(id="vertices", type=VFTYPE),
                gtir.Sym(id="nvertices", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=vertex_domain,
                    target=gtir.SymRef(id="vertices"),
                )
            ],
        )
        sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH.offset_provider)

        # new empty output field
        v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            edges=e,
            vertices=v,
            connectivity_V2E=connectivity_V2E.table,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_with_skip_values():
    init_value = np.random.rand()
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
    )
    stencil_inlined = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                    im.neighbors("V2E", "it")
                )
            ),
            vertex_domain,
        )
    )("edges")
    stencil_fieldview = im.call(
        im.call("as_fieldop")(
            im.lambda_("it")(
                im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                    im.deref("it")
                )
            ),
            vertex_domain,
        )
    )(
        im.call(
            im.call("as_fieldop")(
                im.lambda_("it")(im.neighbors("V2E", "it")),
                vertex_domain,
            )
        )("edges")
    )

    connectivity_V2E = SKIP_VALUE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SKIP_VALUE_MESH.num_edges)
    v_ref = [
        reduce(lambda x, y: x + y, [e[i] if i != -1 else 0.0 for i in v2e_neighbors], init_value)
        for v2e_neighbors in connectivity_V2E.table
    ]

    for i, stencil in enumerate([stencil_inlined, stencil_fieldview]):
        testee = gtir.Program(
            id=f"reduce_with_skip_values_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="edges", type=EFTYPE),
                gtir.Sym(id="vertices", type=VFTYPE),
                gtir.Sym(id="nvertices", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=vertex_domain,
                    target=gtir.SymRef(id="vertices"),
                )
            ],
        )
        sdfg = dace_backend.build_sdfg_from_gtir(testee, SKIP_VALUE_MESH.offset_provider)

        # new empty output field
        v = np.empty(SKIP_VALUE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            edges=e,
            vertices=v,
            connectivity_V2E=connectivity_V2E.table,
            **FSYMBOLS,
            **make_mesh_symbols(SKIP_VALUE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_dot_product():
    init_value = np.random.rand()
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
    )
    v2e_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
        im.call("named_range")(
            gtir.AxisLiteral(value=V2EDim.value, kind=V2EDim.kind),
            0,
            SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
        ),
    )

    testee = gtir.Program(
        id=f"reduce_dot_product",
        function_definitions=[],
        params=[
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
            gtir.Sym(id="nedges", type=SIZE_TYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(
                            im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                                im.deref("it")
                            )
                        ),
                        vertex_domain,
                    )
                )(
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("a", "b")(im.multiplies_(im.deref("a"), im.deref("b"))),
                            v2e_domain,
                        )
                    )(
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("it")(im.neighbors("V2E", "it")),
                                vertex_domain,
                            )
                        )("edges"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("it")(im.plus(im.deref("it"), 1)),
                                v2e_domain,
                            )
                        )(
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("it")(im.neighbors("V2E", "it")),
                                    vertex_domain,
                                )
                            )("edges")
                        ),
                    ),
                ),
                domain=vertex_domain,
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH.offset_provider)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)
    v_ref = [
        reduce(lambda x, y: x + y, e[v2e_neighbors] * (e[v2e_neighbors] + 1), init_value)
        for v2e_neighbors in connectivity_V2E.table
    ]

    sdfg(
        edges=e,
        vertices=v,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
    )
    assert np.allclose(v, v_ref)


def test_gtir_reduce_with_cond_neighbors():
    init_value = np.random.rand()
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(gtir.AxisLiteral(value=Vertex.value), 0, "nvertices"),
    )
    testee = gtir.Program(
        id=f"reduce_with_cond_neighbors",
        function_definitions=[],
        params=[
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(
                            im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                                im.deref("it")
                            )
                        ),
                        vertex_domain,
                    )
                )(
                    im.call(
                        im.call("cond")(
                            im.deref("pred"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("it")(im.neighbors("V2E_FULL", "it")),
                                    vertex_domain,
                                )
                            )("edges"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("it")(im.neighbors("V2E", "it")),
                                    vertex_domain,
                                )
                            )("edges"),
                        ),
                    )()
                ),
                domain=vertex_domain,
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    connectivity_V2E_simple = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E_simple, gtx_common.NeighborTable)
    connectivity_V2E_skip_values = copy.deepcopy(SKIP_VALUE_MESH.offset_provider["V2E"])
    assert isinstance(connectivity_V2E_skip_values, gtx_common.NeighborTable)
    assert SKIP_VALUE_MESH.num_vertices <= SIMPLE_MESH.num_vertices
    connectivity_V2E_skip_values.table = np.concatenate(
        (
            connectivity_V2E_skip_values.table[:, 0 : connectivity_V2E_simple.max_neighbors],
            connectivity_V2E_simple.table[SKIP_VALUE_MESH.num_vertices :, :],
        ),
        axis=0,
    )
    connectivity_V2E_skip_values.max_neighbors = connectivity_V2E_simple.max_neighbors

    e = np.random.rand(SIMPLE_MESH.num_edges)

    for use_full in [False, True]:
        sdfg = dace_backend.build_sdfg_from_gtir(
            testee,
            SIMPLE_MESH.offset_provider | {"V2E_FULL": connectivity_V2E_skip_values},
        )

        v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)
        v_ref = [
            reduce(
                lambda x, y: x + y, [e[i] if i != -1 else 0.0 for i in v2e_neighbors], init_value
            )
            for v2e_neighbors in (
                connectivity_V2E_simple.table if use_full else connectivity_V2E_skip_values.table
            )
        ]
        sdfg(
            pred=np.bool_(use_full),
            edges=e,
            vertices=v,
            connectivity_V2E=connectivity_V2E_skip_values.table,
            connectivity_V2E_FULL=connectivity_V2E_simple.table,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
            __connectivity_V2E_FULL_size_0=SIMPLE_MESH.num_edges,
            __connectivity_V2E_FULL_size_1=connectivity_V2E_skip_values.max_neighbors,
            __connectivity_V2E_FULL_stride_0=connectivity_V2E_skip_values.max_neighbors,
            __connectivity_V2E_FULL_stride_1=1,
        )
        assert np.allclose(v, v_ref)
