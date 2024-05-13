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

from typing import Union
from gt4py.next.common import Connectivity, Dimension, NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace_fieldview.gtir_to_sdfg import (
    GTIRToSDFG as FieldviewGtirToSDFG,
)
from gt4py.next.type_system import type_specifications as ts

import numpy as np

import pytest

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    V2E,
    Edge,
    IDim,
    MeshDescriptor,
    Vertex,
    simple_mesh,
)
from next_tests.integration_tests.cases import EField, IFloatField, VField

dace = pytest.importorskip("dace")


N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
EFTYPE = ts.FieldType(dims=[Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
VFTYPE = ts.FieldType(dims=[Vertex], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
SIMPLE_MESH: MeshDescriptor = simple_mesh()
FSYMBOLS = dict(
    __edges_size_0=SIMPLE_MESH.num_edges,
    __edges_stride_0=1,
    __vertices_size_0=SIMPLE_MESH.num_vertices,
    __vertices_stride_0=1,
    nedges=SIMPLE_MESH.num_edges,
    nvertices=SIMPLE_MESH.num_vertices,
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

    sdfg_genenerator = FieldviewGtirToSDFG(
        [IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)], offset_provider={}
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, y=b, **FSYMBOLS)
    assert np.allclose(a, b)


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

    sdfg_genenerator = FieldviewGtirToSDFG(
        [IFTYPE, IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)], offset_provider={}
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

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

    sdfg_genenerator = FieldviewGtirToSDFG(
        [IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)],
        offset_provider={},
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, z=b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee_fieldview = itir.Program(
        id="sum_3fields",
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
                expr=im.call(
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
                ),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )
    testee_inlined = itir.Program(
        id="sum_3fields",
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
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b", "c")(
                            im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c")))
                        ),
                        domain,
                    )
                )("x", "y", "w"),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)
    d = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [IFTYPE, IFTYPE, IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)],
        offset_provider={},
    )

    for testee in [testee_fieldview, testee_inlined]:
        sdfg = sdfg_genenerator.visit(testee)
        assert isinstance(sdfg, dace.SDFG)

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
    d = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [
            IFTYPE,
            IFTYPE,
            IFTYPE,
            IFTYPE,
            ts.ScalarType(ts.ScalarKind.BOOL),
            ts.ScalarType(ts.ScalarKind.FLOAT64),
            ts.ScalarType(ts.ScalarKind.INT32),
        ],
        offset_provider={},
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    for s in [False, True]:
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
    b = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [
            IFTYPE,
            IFTYPE,
            ts.ScalarType(ts.ScalarKind.BOOL),
            ts.ScalarType(ts.ScalarKind.BOOL),
            ts.ScalarType(ts.ScalarKind.INT32),
        ],
        offset_provider={},
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    for s1 in [False, True]:
        for s2 in [False, True]:
            sdfg(cond_1=np.bool_(s1), cond_2=np.bool_(s2), x=a, z=b, **FSYMBOLS)
            assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))


def test_gtir_cartesian_shift():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="caresian_shift",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.plus(im.deref(im.shift("IDim", 1)("a")), 1)),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N + 1)
    b = np.empty(N)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [IFTYPE, IFTYPE, ts.ScalarType(ts.ScalarKind.INT32)], offset_provider={"IDim": IDim}
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    FSYMBOLS_tmp = FSYMBOLS.copy()
    FSYMBOLS_tmp["__x_size_0"] = N + 1
    sdfg(x=a, y=b, **FSYMBOLS_tmp)
    assert np.allclose(a[1:] + 1, b)


def test_gtir_connectivity_shift():
    vertex_domain = im.call("unstructured_domain")(
        im.call("named_range")(itir.AxisLiteral(value=Vertex.value), 0, "nvertices")
    )
    testee = itir.Program(
        id="connectivity_shift",
        function_definitions=[],
        params=[
            itir.Sym(id="edges"),
            itir.Sym(id="vertices"),
            itir.Sym(id="nedges"),
            itir.Sym(id="nvertices"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(im.deref(im.shift("V2E", 1)("it"))),
                        vertex_domain,
                    )
                )("edges"),
                domain=vertex_domain,
                target=itir.SymRef(id="vertices"),
            )
        ],
    )

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v = np.empty(SIMPLE_MESH.num_vertices)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [EFTYPE, VFTYPE, ts.ScalarType(ts.ScalarKind.INT32), ts.ScalarType(ts.ScalarKind.INT32)],
        offset_provider=SIMPLE_MESH.offset_provider,
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)
    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]
    assert isinstance(connectivity_V2E, NeighborTable)

    sdfg(
        edges=e,
        vertices=v,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        __connectivity_V2E_size_0=SIMPLE_MESH.num_vertices,
        __connectivity_V2E_size_1=SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
        __connectivity_V2E_stride_0=SIMPLE_MESH.offset_provider["V2E"].max_neighbors,
        __connectivity_V2E_stride_1=1,
    )
    assert np.allclose(v, e[connectivity_V2E.table[:, 1]])
