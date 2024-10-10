# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test that ITIR can be lowered to SDFG.

Note: this test module covers the fieldview flavour of ITIR.
"""

import copy
import functools

import numpy as np
import pytest

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import ir_makers as im
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

from . import pytestmark

dace_backend = pytest.importorskip("gt4py.next.program_processors.runners.dace_fieldview")


N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
CFTYPE = ts.FieldType(dims=[Cell], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
EFTYPE = ts.FieldType(dims=[Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
VFTYPE = ts.FieldType(dims=[Vertex], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
V2E_FTYPE = ts.FieldType(dims=[Vertex, V2EDim], dtype=EFTYPE.dtype)
CARTESIAN_OFFSETS = {
    "IDim": IDim,
}
SIMPLE_MESH: MeshDescriptor = simple_mesh()
SIMPLE_MESH_OFFSET_PROVIDER: dict[str, gtx_common.Connectivity | gtx_common.Dimension] = (
    SIMPLE_MESH.offset_provider | CARTESIAN_OFFSETS
)
SKIP_VALUE_MESH: MeshDescriptor = skip_value_mesh()
SKIP_VALUE_MESH_OFFSET_PROVIDER: dict[str, gtx_common.Connectivity | gtx_common.Dimension] = (
    SKIP_VALUE_MESH.offset_provider | CARTESIAN_OFFSETS
)
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
        __connectivity_C2E_size_0=mesh.num_cells,
        __connectivity_C2E_size_1=mesh.offset_provider["C2E"].max_neighbors,
        __connectivity_C2E_stride_0=mesh.offset_provider["C2E"].max_neighbors,
        __connectivity_C2E_stride_1=1,
        __connectivity_C2V_size_0=mesh.num_cells,
        __connectivity_C2V_size_1=mesh.offset_provider["C2V"].max_neighbors,
        __connectivity_C2V_stride_0=mesh.offset_provider["C2V"].max_neighbors,
        __connectivity_C2V_stride_1=1,
        __connectivity_E2V_size_0=mesh.num_edges,
        __connectivity_E2V_size_1=mesh.offset_provider["E2V"].max_neighbors,
        __connectivity_E2V_stride_0=mesh.offset_provider["E2V"].max_neighbors,
        __connectivity_E2V_stride_1=1,
        __connectivity_V2E_size_0=mesh.num_vertices,
        __connectivity_V2E_size_1=mesh.offset_provider["V2E"].max_neighbors,
        __connectivity_V2E_stride_0=mesh.offset_provider["V2E"].max_neighbors,
        __connectivity_V2E_stride_1=1,
    )


def test_gtir_broadcast():
    val = np.random.rand()
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_broadcast",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop("deref", domain)(val),
                domain=domain,
                target=gtir.SymRef(id="x"),
            )
        ],
    )

    a = np.empty(N, dtype=np.float64)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, **FSYMBOLS)
    np.testing.assert_array_equal(a, val)


def test_gtir_cast():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    IFTYPE_FLOAT32 = ts.FieldType(IFTYPE.dims, dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    IFTYPE_BOOL = ts.FieldType(IFTYPE.dims, dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL))
    testee = gtir.Program(
        id="gtir_cast",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE_FLOAT32),
            gtir.Sym(id="z", type=IFTYPE_BOOL),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("eq", domain)(
                    im.as_fieldop(
                        im.lambda_("a")(im.call("cast_")(im.deref("a"), "float32")), domain
                    )("x"),
                    "y",
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.ones(N, dtype=np.float64) * np.sqrt(2.0)
    b = a.astype(np.float32)
    c = np.empty_like(a, dtype=np.bool_)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    np.testing.assert_array_equal(c, True)


def test_gtir_copy_self():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (1, 2)})
    testee = gtir.Program(
        id="gtir_copy_self",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=gtir.SymRef(id="x"),
                domain=domain,
                target=gtir.SymRef(id="x"),
            )
        ],
    )

    a = np.random.rand(N)
    ref = a.copy()

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, **FSYMBOLS)
    assert np.allclose(a, ref)


def test_gtir_tuple_swap():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_tuple_swap",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple("y", "x"),
                domain=domain,
                # TODO(havogt): add a frontend check for this pattern
                target=im.make_tuple("x", "y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = (a.copy(), b.copy())

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(a, ref[1])
    assert np.allclose(b, ref[0])


def test_gtir_tuple_args():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_tuple_args",
        function_definitions=[],
        params=[
            gtir.Sym(
                id="x", type=ts.TupleType(types=[IFTYPE, ts.TupleType(types=[IFTYPE, IFTYPE])])
            ),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus", domain)(
                    im.tuple_get(0, "x"),
                    im.op_as_fieldop("plus", domain)(
                        im.tuple_get(0, im.tuple_get(1, "x")),
                        im.tuple_get(1, im.tuple_get(1, "x")),
                    ),
                ),
                domain=domain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    x_fields = (a, a, b)
    x_symbols = dict(
        __x_0_size_0=FSYMBOLS["__x_size_0"],
        __x_0_stride_0=FSYMBOLS["__x_stride_0"],
        __x_1_0_size_0=FSYMBOLS["__x_size_0"],
        __x_1_0_stride_0=FSYMBOLS["__x_stride_0"],
        __x_1_1_size_0=FSYMBOLS["__y_size_0"],
        __x_1_1_stride_0=FSYMBOLS["__y_stride_0"],
    )

    sdfg(*x_fields, c, **FSYMBOLS, **x_symbols)
    assert np.allclose(c, a * 2 + b)


def test_gtir_tuple_expr():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_tuple_expr",
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
                expr=im.op_as_fieldop("plus", domain)(
                    im.tuple_get(0, im.make_tuple("x", im.make_tuple("x", "y"))),
                    im.op_as_fieldop("plus", domain)(
                        im.tuple_get(
                            0, im.tuple_get(1, im.make_tuple("x", im.make_tuple("x", "y")))
                        ),
                        im.tuple_get(
                            1, im.tuple_get(1, im.make_tuple("x", im.make_tuple("x", "y")))
                        ),
                    ),
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, a * 2 + b)


def test_gtir_tuple_return():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_tuple_return",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(
                id="z", type=ts.TupleType(types=[ts.TupleType(types=[IFTYPE, IFTYPE]), IFTYPE])
            ),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple(
                    im.make_tuple(im.op_as_fieldop("plus", domain)("x", "y"), "x"), "y"
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    z_fields = (np.empty_like(a), np.empty_like(a), np.empty_like(a))
    z_symbols = dict(
        __z_0_0_size_0=FSYMBOLS["__x_size_0"],
        __z_0_0_stride_0=FSYMBOLS["__x_stride_0"],
        __z_0_1_size_0=FSYMBOLS["__x_size_0"],
        __z_0_1_stride_0=FSYMBOLS["__x_stride_0"],
        __z_1_size_0=FSYMBOLS["__x_size_0"],
        __z_1_stride_0=FSYMBOLS["__x_stride_0"],
    )

    sdfg(a, b, *z_fields, **FSYMBOLS, **z_symbols)
    assert np.allclose(z_fields[0], a + b)
    assert np.allclose(z_fields[1], a)
    assert np.allclose(z_fields[2], b)


def test_gtir_tuple_target():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="gtir_tuple_target",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple(im.op_as_fieldop("plus", domain)("x", 1.0), gtir.SymRef(id="x")),
                domain=domain,
                target=im.make_tuple("x", "y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)
    ref = a.copy()

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(a, ref + 1)
    assert np.allclose(b, ref)


def test_gtir_update():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    stencil1 = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref("a"), im.plus(im.minus(0.0, 2.0), 1.0))),
        domain,
    )("x")
    stencil2 = im.op_as_fieldop("plus", domain)("x", im.plus(im.minus(0.0, 2.0), 1.0))

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
        ref = a - 1.0

        sdfg(a, **FSYMBOLS)
        assert np.allclose(a, ref)


def test_gtir_sum2():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
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
                expr=im.op_as_fieldop("plus", domain)("x", "y"),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
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
                expr=im.op_as_fieldop("plus", domain)("x", "x"),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    stencil1 = im.op_as_fieldop("plus", domain)(
        "x",
        im.op_as_fieldop("plus", domain)("y", "w"),
    )
    stencil2 = im.as_fieldop(
        im.lambda_("a", "b", "c")(im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c")))),
        domain,
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

        sdfg(a, b, c, d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


def test_gtir_cond():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="cond_2sums",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="w", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="s1", type=ts.ScalarType(ts.ScalarKind.INT32)),
            gtir.Sym(id="s2", type=ts.ScalarType(ts.ScalarKind.INT32)),
            gtir.Sym(id="scalar", type=ts.ScalarType(ts.ScalarKind.FLOAT64)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus", domain)(
                    "x",
                    im.if_(
                        im.greater(gtir.SymRef(id="s1"), gtir.SymRef(id="s2")),
                        im.op_as_fieldop("plus", domain)("y", "scalar"),
                        im.op_as_fieldop("plus", domain)("w", "scalar"),
                    ),
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    for s1, s2 in [(1, 2), (2, 1)]:
        d = np.empty_like(a)
        sdfg(a, b, c, d, s1, s2, scalar=1.0, **FSYMBOLS)
        assert np.allclose(d, (a + b + 1) if s1 > s2 else (a + c + 1))


def test_gtir_cond_with_tuple_return():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="cond_with_tuple_return",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="w", type=IFTYPE),
            gtir.Sym(id="z", type=ts.TupleType(types=[IFTYPE, IFTYPE])),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.tuple_get(
                    0,
                    im.if_(
                        gtir.SymRef(id="pred"),
                        im.make_tuple(im.make_tuple("x", "y"), "w"),
                        im.make_tuple(im.make_tuple("y", "x"), "w"),
                    ),
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    z_symbols = dict(
        __z_0_size_0=FSYMBOLS["__x_size_0"],
        __z_0_stride_0=FSYMBOLS["__x_stride_0"],
        __z_1_size_0=FSYMBOLS["__x_size_0"],
        __z_1_stride_0=FSYMBOLS["__x_stride_0"],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    for s in [False, True]:
        z_fields = (np.empty_like(a), np.empty_like(a))
        sdfg(a, b, c, *z_fields, pred=np.bool_(s), **FSYMBOLS, **z_symbols)
        assert np.allclose(z_fields[0], a if s else b)
        assert np.allclose(z_fields[1], b if s else a)


def test_gtir_cond_nested():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
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
                expr=im.if_(
                    gtir.SymRef(id="pred_1"),
                    im.op_as_fieldop("plus", domain)("x", 1.0),
                    im.if_(
                        gtir.SymRef(id="pred_2"),
                        im.op_as_fieldop("plus", domain)("x", 2.0),
                        im.op_as_fieldop("plus", domain)("x", 3.0),
                    ),
                ),
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
            sdfg(a, b, pred_1=np.bool_(s1), pred_2=np.bool_(s2), **FSYMBOLS)
            assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))


def test_gtir_cartesian_shift_left():
    DELTA = 3.0
    OFFSET = 1
    domain = im.domain(
        gtx_common.GridType.CARTESIAN,
        ranges={
            IDim: (
                0,
                im.minus(gtir.SymRef(id="size"), gtir.Literal(value=str(OFFSET), type=SIZE_TYPE)),
            ),
        },
    )

    # cartesian shift with literal integer offset
    stencil1_inlined = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref(im.shift("IDim", OFFSET)("a")), DELTA)),
        domain,
    )("x")
    # fieldview flavor of same stencil, in which a temporary field is initialized with the `DELTA` constant value
    stencil1_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a")(im.deref(im.shift("IDim", OFFSET)("a"))),
            domain,
        )("x"),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    # use dynamic offset retrieved from field
    stencil2_inlined = im.as_fieldop(
        im.lambda_("a", "off")(im.plus(im.deref(im.shift("IDim", im.deref("off"))("a")), DELTA)),
        domain,
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil2_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a", "off")(im.deref(im.shift("IDim", im.deref("off"))("a"))),
            domain,
        )("x", "x_offset"),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    # use the result of an arithmetic field operation as dynamic offset
    stencil3_inlined = im.as_fieldop(
        im.lambda_("a", "off")(
            im.plus(im.deref(im.shift("IDim", im.plus(im.deref("off"), 0))("a")), DELTA)
        ),
        domain,
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil3_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a", "off")(im.deref(im.shift("IDim", im.deref("off"))("a"))),
            domain,
        )(
            "x",
            im.op_as_fieldop("plus", domain)("x_offset", 0),
        ),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    a = np.random.rand(N)
    a_offset = np.full(N, OFFSET, dtype=np.int32)
    b = np.empty(N)

    IOFFSET_FTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))

    for i, stencil in enumerate(
        [
            stencil1_inlined,
            stencil1_fieldview,
            stencil2_inlined,
            stencil2_fieldview,
            stencil3_inlined,
            stencil3_fieldview,
        ]
    ):
        testee = gtir.Program(
            id=f"cartesian_shift_left_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=IFTYPE),
                gtir.Sym(id="x_offset", type=IOFFSET_FTYPE),
                gtir.Sym(id="y", type=IFTYPE),
                gtir.Sym(id="size", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=gtir.SymRef(id="y"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

        FSYMBOLS_tmp = FSYMBOLS.copy()
        FSYMBOLS_tmp["__x_offset_stride_0"] = 1
        sdfg(a, a_offset, b, **FSYMBOLS_tmp)
        assert np.allclose(a[OFFSET:] + DELTA, b[:-OFFSET])


def test_gtir_cartesian_shift_right():
    DELTA = 3.0
    OFFSET = 1
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (OFFSET, "size")})

    # cartesian shift with literal integer offset
    stencil1_inlined = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref(im.shift("IDim", -OFFSET)("a")), DELTA)),
        domain,
    )("x")
    # fieldview flavor of same stencil, in which a temporary field is initialized with the `DELTA` constant value
    stencil1_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a")(im.deref(im.shift("IDim", -OFFSET)("a"))),
            domain,
        )("x"),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    # use dynamic offset retrieved from field
    stencil2_inlined = im.as_fieldop(
        im.lambda_("a", "off")(im.plus(im.deref(im.shift("IDim", im.deref("off"))("a")), DELTA)),
        domain,
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil2_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a", "off")(im.deref(im.shift("IDim", im.deref("off"))("a"))),
            domain,
        )("x", "x_offset"),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    # use the result of an arithmetic field operation as dynamic offset
    stencil3_inlined = im.as_fieldop(
        im.lambda_("a", "off")(
            im.plus(im.deref(im.shift("IDim", im.plus(im.deref("off"), 0))("a")), DELTA)
        ),
        domain,
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil3_fieldview = im.op_as_fieldop("plus", domain)(
        im.as_fieldop(
            im.lambda_("a", "off")(im.deref(im.shift("IDim", im.deref("off"))("a"))),
            domain,
        )(
            "x",
            im.op_as_fieldop("plus", domain)("x_offset", 0),
        ),
        im.as_fieldop(im.lambda_()(DELTA), domain)(),
    )

    a = np.random.rand(N)
    a_offset = np.full(N, -OFFSET, dtype=np.int32)
    b = np.empty(N)

    IOFFSET_FTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))

    for i, stencil in enumerate(
        [
            stencil1_inlined,
            stencil1_fieldview,
            stencil2_inlined,
            stencil2_fieldview,
            stencil3_inlined,
            stencil3_fieldview,
        ]
    ):
        testee = gtir.Program(
            id=f"cartesian_shift_right_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=IFTYPE),
                gtir.Sym(id="x_offset", type=IOFFSET_FTYPE),
                gtir.Sym(id="y", type=IFTYPE),
                gtir.Sym(id="size", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=domain,
                    target=gtir.SymRef(id="y"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

        sdfg(a, a_offset, b, **FSYMBOLS, __x_offset_stride_0=1)
        assert np.allclose(a[:-OFFSET] + DELTA, b[OFFSET:])


def test_gtir_connectivity_shift():
    C2E_neighbor_idx = 2
    E2V_neighbor_idx = 1
    edge_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Edge: (0, "nedges")})
    ce_domain = im.domain(
        gtx_common.GridType.UNSTRUCTURED,
        ranges={
            Cell: (0, "ncells"),
            Edge: (0, "nedges"),
        },
    )
    cv_domain = im.domain(
        gtx_common.GridType.UNSTRUCTURED,
        ranges={
            Cell: (0, "ncells"),
            Vertex: (0, "nvertices"),
        },
    )

    # apply shift 2 times along different dimensions
    stencil1_inlined = im.as_fieldop(
        im.lambda_("it")(
            im.deref(im.shift("C2E", C2E_neighbor_idx)(im.shift("E2V", E2V_neighbor_idx)("it")))
        ),
        ce_domain,
    )("ev_field")

    # fieldview flavor of the same stncil: create an intermediate temporary field
    stencil1_fieldview = im.as_fieldop(
        im.lambda_("it")(im.deref(im.shift("E2V", E2V_neighbor_idx)("it"))),
        ce_domain,
    )(
        im.as_fieldop(
            im.lambda_("it")(im.deref(im.shift("C2E", C2E_neighbor_idx)("it"))),
            cv_domain,
        )("ev_field")
    )

    # multi-dimensional shift in one function call
    stencil2 = im.as_fieldop(
        im.lambda_("it")(
            im.deref(
                im.call(
                    im.call("shift")(
                        im.ensure_offset("E2V"),
                        im.ensure_offset(E2V_neighbor_idx),
                        im.ensure_offset("C2E"),
                        im.ensure_offset(C2E_neighbor_idx),
                    )
                )("it")
            )
        ),
        ce_domain,
    )("ev_field")

    # again multi-dimensional shift in one function call, but this time with dynamic offset values
    stencil3_inlined = im.as_fieldop(
        im.lambda_("it", "c2e_off", "e2v_off")(
            im.deref(
                im.call(
                    im.call("shift")(
                        im.ensure_offset("E2V"),
                        im.plus(im.deref("e2v_off"), 0),
                        im.ensure_offset("C2E"),
                        im.deref("c2e_off"),
                    )
                )("it")
            )
        ),
        ce_domain,
    )("ev_field", "c2e_offset", "e2v_offset")

    # fieldview flavor of same stencil with dynamic offset
    stencil3_fieldview = im.as_fieldop(
        im.lambda_("it", "c2e_off", "e2v_off")(
            im.deref(
                im.call(
                    im.call("shift")(
                        im.ensure_offset("E2V"),
                        im.deref("e2v_off"),
                        im.ensure_offset("C2E"),
                        im.deref("c2e_off"),
                    )
                )("it")
            )
        ),
        ce_domain,
    )(
        "ev_field",
        "c2e_offset",
        im.op_as_fieldop("plus", edge_domain)("e2v_offset", 0),
    )

    CE_FTYPE = ts.FieldType(dims=[Cell, Edge], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    EV_FTYPE = ts.FieldType(dims=[Edge, Vertex], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    CELL_OFFSET_FTYPE = ts.FieldType(dims=[Cell], dtype=SIZE_TYPE)
    EDGE_OFFSET_FTYPE = ts.FieldType(dims=[Edge], dtype=SIZE_TYPE)

    connectivity_C2E = SIMPLE_MESH_OFFSET_PROVIDER["C2E"]
    assert isinstance(connectivity_C2E, gtx_common.NeighborTable)
    connectivity_E2V = SIMPLE_MESH_OFFSET_PROVIDER["E2V"]
    assert isinstance(connectivity_E2V, gtx_common.NeighborTable)

    ev = np.random.rand(SIMPLE_MESH.num_edges, SIMPLE_MESH.num_vertices)
    ref = ev[connectivity_C2E.table[:, C2E_neighbor_idx], :][
        :, connectivity_E2V.table[:, E2V_neighbor_idx]
    ]

    for i, stencil in enumerate(
        [stencil1_inlined, stencil1_fieldview, stencil2, stencil3_inlined, stencil3_fieldview]
    ):
        testee = gtir.Program(
            id=f"connectivity_shift_2d_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="ce_field", type=CE_FTYPE),
                gtir.Sym(id="ev_field", type=EV_FTYPE),
                gtir.Sym(id="c2e_offset", type=CELL_OFFSET_FTYPE),
                gtir.Sym(id="e2v_offset", type=EDGE_OFFSET_FTYPE),
                gtir.Sym(id="ncells", type=SIZE_TYPE),
                gtir.Sym(id="nedges", type=SIZE_TYPE),
                gtir.Sym(id="nvertices", type=SIZE_TYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=ce_domain,
                    target=gtir.SymRef(id="ce_field"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

        ce = np.empty([SIMPLE_MESH.num_cells, SIMPLE_MESH.num_edges])

        sdfg(
            ce,
            ev,
            c2e_offset=np.full(SIMPLE_MESH.num_cells, C2E_neighbor_idx, dtype=np.int32),
            e2v_offset=np.full(SIMPLE_MESH.num_edges, E2V_neighbor_idx, dtype=np.int32),
            connectivity_C2E=connectivity_C2E.table,
            connectivity_E2V=connectivity_E2V.table,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
            __ce_field_size_0=SIMPLE_MESH.num_cells,
            __ce_field_size_1=SIMPLE_MESH.num_edges,
            __ce_field_stride_0=SIMPLE_MESH.num_edges,
            __ce_field_stride_1=1,
            __ev_field_size_0=SIMPLE_MESH.num_edges,
            __ev_field_size_1=SIMPLE_MESH.num_vertices,
            __ev_field_stride_0=SIMPLE_MESH.num_vertices,
            __ev_field_stride_1=1,
            __c2e_offset_stride_0=1,
            __e2v_offset_stride_0=1,
        )
        assert np.allclose(ce, ref)


def test_gtir_connectivity_shift_chain():
    E2V_neighbor_idx = 1
    V2E_neighbor_idx = 2
    edge_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Edge: (0, "nedges")})
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
    testee = gtir.Program(
        id="connectivity_shift_chain",
        function_definitions=[],
        params=[
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="edges_out", type=EFTYPE),
            gtir.Sym(id="nedges", type=SIZE_TYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.deref(im.shift("E2V", E2V_neighbor_idx)("it"))),
                    edge_domain,
                )(
                    im.as_fieldop(
                        im.lambda_("it")(im.deref(im.shift("V2E", V2E_neighbor_idx)("it"))),
                        vertex_domain,
                    )("edges")
                ),
                domain=edge_domain,
                target=gtir.SymRef(id="edges_out"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

    connectivity_E2V = SIMPLE_MESH_OFFSET_PROVIDER["E2V"]
    assert isinstance(connectivity_E2V, gtx_common.NeighborTable)
    connectivity_V2E = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    ref = e[connectivity_V2E.table[connectivity_E2V.table[:, E2V_neighbor_idx], V2E_neighbor_idx]]

    # new empty output field
    e_out = np.empty_like(e)

    sdfg(
        e,
        e_out,
        connectivity_E2V=connectivity_E2V.table,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
        __edges_out_size_0=SIMPLE_MESH.num_edges,
        __edges_out_stride_0=1,
    )
    assert np.allclose(e_out, ref)


def test_gtir_neighbors_as_input():
    # FIXME[#1582](edopao): Enable testcase when type inference is working
    pytest.skip("Field of lists not fully supported by GTIR type inference")
    init_value = np.random.rand()
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
    testee = gtir.Program(
        id="neighbors_as_input",
        function_definitions=[],
        params=[
            gtir.Sym(id="v2e_field", type=V2E_FTYPE),
            gtir.Sym(id="vertex", type=EFTYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("it")(
                            im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                                "it"
                            )
                        ),
                        vertex_domain,
                    )
                )("v2e_field"),
                domain=vertex_domain,
                target=gtir.SymRef(id="vertex"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

    connectivity_V2E = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    v2e_field = np.random.rand(SIMPLE_MESH.num_vertices, connectivity_V2E.max_neighbors)
    v = np.empty(SIMPLE_MESH.num_vertices, dtype=v2e_field.dtype)

    v_ref = [
        functools.reduce(lambda x, y: x + y, v2e_neighbors, init_value)
        for v2e_neighbors in v2e_field
    ]

    sdfg(
        v2e_field,
        v,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
        __v2e_field_size_0=SIMPLE_MESH.num_vertices,
        __v2e_field_size_1=connectivity_V2E.max_neighbors,
        __v2e_field_stride_0=connectivity_V2E.max_neighbors,
        __v2e_field_stride_1=1,
    )
    assert np.allclose(v, v_ref)


def test_gtir_neighbors_as_output():
    # FIXME[#1582](edopao): Enable testcase when type inference is working
    pytest.skip("Field of lists not fully supported by GTIR type inference")
    v2e_domain = im.domain(
        gtx_common.GridType.UNSTRUCTURED,
        ranges={
            Vertex: (0, "nvertices"),
            V2EDim: (0, SIMPLE_MESH_OFFSET_PROVIDER["V2E"].max_neighbors),
        },
    )
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
    testee = gtir.Program(
        id="neighbors_as_output",
        function_definitions=[],
        params=[
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="v2e_field", type=V2E_FTYPE),
            gtir.Sym(id="nvertices", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop_neighbors("V2E", "edges", vertex_domain),
                domain=v2e_domain,
                target=gtir.SymRef(id="v2e_field"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

    connectivity_V2E = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v2e_field = np.empty([SIMPLE_MESH.num_vertices, connectivity_V2E.max_neighbors], dtype=e.dtype)

    sdfg(
        e,
        v2e_field,
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
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
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
    )(im.as_fieldop_neighbors("V2E", "edges", vertex_domain))

    connectivity_V2E = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v_ref = [
        functools.reduce(lambda x, y: x + y, e[v2e_neighbors], init_value)
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
        sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

        # new empty output field
        v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            e,
            v,
            connectivity_V2E=connectivity_V2E.table,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_with_skip_values():
    init_value = np.random.rand()
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
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
    )(im.as_fieldop_neighbors("V2E", "edges", vertex_domain))

    connectivity_V2E = SKIP_VALUE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    e = np.random.rand(SKIP_VALUE_MESH.num_edges)
    v_ref = [
        functools.reduce(
            lambda x, y: x + y, [e[i] if i != -1 else 0.0 for i in v2e_neighbors], init_value
        )
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
        sdfg = dace_backend.build_sdfg_from_gtir(testee, SKIP_VALUE_MESH_OFFSET_PROVIDER)

        # new empty output field
        v = np.empty(SKIP_VALUE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            e,
            v,
            connectivity_V2E=connectivity_V2E.table,
            **FSYMBOLS,
            **make_mesh_symbols(SKIP_VALUE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_dot_product():
    # FIXME[#1582](edopao): Enable testcase when type inference is working
    pytest.skip("Field of lists not fully supported as a type in GTIR yet")
    init_value = np.random.rand()
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})

    testee = gtir.Program(
        id="reduce_dot_product",
        function_definitions=[],
        params=[
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
                    im.op_as_fieldop("multiplies", vertex_domain)(
                        im.as_fieldop_neighbors("V2E", "edges", vertex_domain),
                        im.as_fieldop_neighbors("V2E", "edges", vertex_domain),
                    ),
                ),
                domain=vertex_domain,
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    connectivity_V2E = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E, gtx_common.NeighborTable)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)
    v_ref = [
        reduce(lambda x, y: x + y, e[v2e_neighbors] * e[v2e_neighbors], init_value)
        for v2e_neighbors in connectivity_V2E.table
    ]

    sdfg(
        e,
        v,
        connectivity_V2E=connectivity_V2E.table,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
    )
    assert np.allclose(v, v_ref)


def test_gtir_reduce_with_cond_neighbors():
    init_value = np.random.rand()
    vertex_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Vertex: (0, "nvertices")})
    testee = gtir.Program(
        id="reduce_with_cond_neighbors",
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
                expr=im.as_fieldop(
                    im.lambda_("it")(
                        im.call(im.call("reduce")("plus", im.literal_from_value(init_value)))(
                            im.deref("it")
                        )
                    ),
                    vertex_domain,
                )(
                    im.if_(
                        gtir.SymRef(id="pred"),
                        im.as_fieldop_neighbors("V2E_FULL", "edges", vertex_domain),
                        im.as_fieldop_neighbors("V2E", "edges", vertex_domain),
                    )
                ),
                domain=vertex_domain,
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    connectivity_V2E_simple = SIMPLE_MESH_OFFSET_PROVIDER["V2E"]
    assert isinstance(connectivity_V2E_simple, gtx_common.NeighborTable)
    connectivity_V2E_skip_values = copy.deepcopy(SKIP_VALUE_MESH_OFFSET_PROVIDER["V2E"])
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
            SIMPLE_MESH_OFFSET_PROVIDER | {"V2E_FULL": connectivity_V2E_skip_values},
        )

        v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)
        v_ref = [
            functools.reduce(
                lambda x, y: x + y, [e[i] if i != -1 else 0.0 for i in v2e_neighbors], init_value
            )
            for v2e_neighbors in (
                connectivity_V2E_simple.table if use_full else connectivity_V2E_skip_values.table
            )
        ]
        sdfg(
            np.bool_(use_full),
            e,
            v,
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


def test_gtir_let_lambda():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    subdomain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (1, im.minus("size", 1))})
    testee = gtir.Program(
        id="let_lambda",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                # `x1` is a let-lambda expression representing `x * 3`
                # `x2` is a let-lambda expression representing `x * 4`
                #  - note that the let-symbol `x2` is used twice, in a nested let-expression, to test aliasing of the symbol
                # `x3` is a let-lambda expression simply accessing `x` field symref
                expr=im.let("x1", im.op_as_fieldop("multiplies", subdomain)(3.0, "x"))(
                    im.let(
                        "x2",
                        im.let("x2", im.op_as_fieldop("multiplies", domain)(2.0, "x"))(
                            im.op_as_fieldop("plus", subdomain)("x2", "x2")
                        ),
                    )(
                        im.let("x3", "x")(
                            im.op_as_fieldop("plus", subdomain)(
                                "x1", im.op_as_fieldop("plus", subdomain)("x2", "x3")
                            )
                        )
                    )
                ),
                domain=subdomain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = np.concatenate((b[0:1], a[1 : N - 1] * 8, b[N - 1 : N]))

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(b, ref)


def test_gtir_let_lambda_with_connectivity():
    C2E_neighbor_idx = 1
    C2V_neighbor_idx = 2
    cell_domain = im.domain(gtx_common.GridType.UNSTRUCTURED, ranges={Cell: (0, "ncells")})

    connectivity_C2E = SIMPLE_MESH_OFFSET_PROVIDER["C2E"]
    assert isinstance(connectivity_C2E, gtx_common.NeighborTable)
    connectivity_C2V = SIMPLE_MESH_OFFSET_PROVIDER["C2V"]
    assert isinstance(connectivity_C2V, gtx_common.NeighborTable)

    testee = gtir.Program(
        id="let_lambda_with_connectivity",
        function_definitions=[],
        params=[
            gtir.Sym(id="cells", type=CFTYPE),
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
            gtir.Sym(id="ncells", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "x1",
                    im.as_fieldop(
                        im.lambda_("it")(im.deref(im.shift("C2E", C2E_neighbor_idx)("it"))),
                        cell_domain,
                    )("edges"),
                )(
                    im.let(
                        "x2",
                        im.as_fieldop(
                            im.lambda_("it")(im.deref(im.shift("C2V", C2V_neighbor_idx)("it"))),
                            cell_domain,
                        )("vertices"),
                    )(im.op_as_fieldop("plus", cell_domain)("x1", "x2"))
                ),
                domain=cell_domain,
                target=gtir.SymRef(id="cells"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, SIMPLE_MESH_OFFSET_PROVIDER)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v = np.random.rand(SIMPLE_MESH.num_vertices)
    c = np.empty(SIMPLE_MESH.num_cells)
    ref = (
        e[connectivity_C2E.table[:, C2E_neighbor_idx]]
        + v[connectivity_C2V.table[:, C2V_neighbor_idx]]
    )

    sdfg(
        cells=c,
        edges=e,
        vertices=v,
        connectivity_C2E=connectivity_C2E.table,
        connectivity_C2V=connectivity_C2V.table,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
    )
    assert np.allclose(c, ref)


def test_gtir_let_lambda_with_cond():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="let_lambda_with_cond",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("x1", "x")(
                    im.let("x2", im.op_as_fieldop("multiplies", domain)(2.0, "x"))(
                        im.if_(
                            gtir.SymRef(id="pred"),
                            im.as_fieldop(im.lambda_("a")(im.deref("a")), domain)("x1"),
                            im.as_fieldop(im.lambda_("a")(im.deref("a")), domain)("x2"),
                        )
                    )
                ),
                domain=domain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})

    a = np.random.rand(N)
    for s in [False, True]:
        b = np.empty_like(a)
        sdfg(a, b, pred=np.bool_(s), **FSYMBOLS)
        assert np.allclose(b, a if s else a * 2)


def test_gtir_let_lambda_with_tuple():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="let_lambda_with_tuple",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=ts.TupleType(types=[IFTYPE, IFTYPE])),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "t",
                    im.make_tuple(
                        im.make_tuple(im.op_as_fieldop("plus", domain)("x", "y"), "x"), "y"
                    ),
                )(im.make_tuple(im.tuple_get(1, im.tuple_get(0, "t")), im.tuple_get(1, "t"))),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    z_fields = (np.empty_like(a), np.empty_like(a))
    z_symbols = dict(
        __z_0_size_0=FSYMBOLS["__x_size_0"],
        __z_0_stride_0=FSYMBOLS["__x_stride_0"],
        __z_1_size_0=FSYMBOLS["__x_size_0"],
        __z_1_stride_0=FSYMBOLS["__x_stride_0"],
    )

    sdfg(a, b, *z_fields, **FSYMBOLS, **z_symbols)
    assert np.allclose(z_fields[0], a)
    assert np.allclose(z_fields[1], b)


def test_gtir_if_scalars():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="if_scalars",
        function_definitions=[],
        params=[
            gtir.Sym(
                id="x",
                type=ts.TupleType(types=[IFTYPE, ts.TupleType(types=[SIZE_TYPE, SIZE_TYPE])]),
            ),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("f", im.tuple_get(0, "x"))(
                    im.let("y", im.tuple_get(1, "x"))(
                        im.let("y_0", im.tuple_get(0, "y"))(
                            im.let("y_1", im.tuple_get(1, "y"))(
                                im.op_as_fieldop("plus", domain)(
                                    "f",
                                    im.if_(
                                        "pred",
                                        im.call("cast_")("y_0", "float64"),
                                        im.call("cast_")("y_1", "float64"),
                                    ),
                                )
                            )
                        )
                    )
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)
    d1 = np.random.randint(0, 1000)
    d2 = np.random.randint(0, 1000)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, {})
    x_symbols = dict(
        __x_0_size_0=FSYMBOLS["__x_size_0"],
        __x_0_stride_0=FSYMBOLS["__x_stride_0"],
    )

    for s in [False, True]:
        sdfg(x_0=a, x_1_0=d1, x_1_1=d2, z=b, pred=np.bool_(s), **FSYMBOLS, **x_symbols)
        assert np.allclose(b, (a + d1 if s else a + d2))


def test_gtir_if_values():
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (0, "size")})
    testee = gtir.Program(
        id="if_values",
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
                expr=im.op_as_fieldop("if_", domain)(
                    im.op_as_fieldop("less", domain)("x", "y"), "x", "y"
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = dace_backend.build_sdfg_from_gtir(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, np.where(a < b, a, b))
