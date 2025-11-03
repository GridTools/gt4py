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

import functools
from typing import Any, Callable

import numpy as np
import pytest

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    Edge,
    IDim,
    JDim,
    KDim,
    MeshDescriptor,
    V2EDim,
    Vertex,
    simple_mesh,
    skip_value_mesh,
)

dace = pytest.importorskip("dace")
dace_backend = pytest.importorskip("gt4py.next.program_processors.runners.dace")


@pytest.fixture
def allow_view_arguments():
    """Allows arrays with non-default layout."""
    path = ["compiler", "allow_view_arguments"]
    value = True
    old_value = dace.Config.get(*path)
    dace.Config.set(*path, value=value)
    yield
    dace.Config.set(*path, value=old_value)


pytestmark = pytest.mark.usefixtures(
    "allow_view_arguments"
)  # use the fixture for all tests in this module

N = 10
BOOL_TYPE = ts.ScalarType(kind=ts.ScalarKind.BOOL)
FLOAT_TYPE = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
IFTYPE = ts.FieldType(dims=[IDim], dtype=FLOAT_TYPE)
CFTYPE = ts.FieldType(dims=[Cell], dtype=FLOAT_TYPE)
EFTYPE = ts.FieldType(dims=[Edge], dtype=FLOAT_TYPE)
VFTYPE = ts.FieldType(dims=[Vertex], dtype=FLOAT_TYPE)
V2E_FTYPE = ts.FieldType(dims=[Vertex, V2EDim], dtype=EFTYPE.dtype)
CARTESIAN_OFFSETS = {
    IDim.value: IDim,
}
SIMPLE_MESH: MeshDescriptor = simple_mesh(None)
SKIP_VALUE_MESH: MeshDescriptor = skip_value_mesh(None)
SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)
FSYMBOLS = dict(
    __w_IDim_range_0=0,
    __w_IDim_range_1=N,
    __w_stride_0=1,
    __x_IDim_range_0=0,
    __x_IDim_range_1=N,
    __x_stride_0=1,
    __y_IDim_range_0=0,
    __y_IDim_range_1=N,
    __y_stride_0=1,
    __z_IDim_range_0=0,
    __z_IDim_range_1=N,
    __z_stride_0=1,
)


def make_mesh_symbols(mesh: MeshDescriptor):
    c2e_ndarray = mesh.offset_provider["C2E"].ndarray
    c2v_ndarray = mesh.offset_provider["C2V"].ndarray
    e2v_ndarray = mesh.offset_provider["E2V"].ndarray
    v2e_ndarray = mesh.offset_provider["V2E"].ndarray
    return dict(
        __cells_Cell_range_0=0,
        __cells_Cell_range_1=mesh.num_cells,
        __cells_stride_0=1,
        __edges_Edge_range_0=0,
        __edges_Edge_range_1=mesh.num_edges,
        __edges_stride_0=1,
        __vertices_Vertex_range_0=0,
        __vertices_Vertex_range_1=mesh.num_vertices,
        __vertices_stride_0=1,
        __gt_conn_C2E_size_0=c2e_ndarray.shape[0],
        __gt_conn_C2E_stride_0=c2e_ndarray.strides[0] // c2e_ndarray.itemsize,
        __gt_conn_C2E_stride_1=c2e_ndarray.strides[1] // c2e_ndarray.itemsize,
        __gt_conn_C2V_size_0=c2v_ndarray.shape[0],
        __gt_conn_C2V_stride_0=c2v_ndarray.strides[0] // c2v_ndarray.itemsize,
        __gt_conn_C2V_stride_1=c2v_ndarray.strides[1] // c2v_ndarray.itemsize,
        __gt_conn_E2V_size_0=e2v_ndarray.shape[0],
        __gt_conn_E2V_stride_0=e2v_ndarray.strides[0] // e2v_ndarray.itemsize,
        __gt_conn_E2V_stride_1=e2v_ndarray.strides[1] // e2v_ndarray.itemsize,
        __gt_conn_V2E_size_0=v2e_ndarray.shape[0],
        __gt_conn_V2E_stride_0=v2e_ndarray.strides[0] // v2e_ndarray.itemsize,
        __gt_conn_V2E_stride_1=v2e_ndarray.strides[1] // v2e_ndarray.itemsize,
    )


def build_dace_sdfg(
    ir: gtir.Program,
    offset_provider: gtx_common.OffsetProvider,
    skip_domain_inference: bool = False,
) -> Callable[..., Any]:
    """Wrapper of `dace_backend.build_sdfg_from_gtir()` to run domain inference.

    Before calling `dace_backend.build_sdfg_from_gtir()`, it will infer the domain
    of the given `ir`, unless called with `skip_domain_inference=True`.
    """
    if not skip_domain_inference:
        # run domain inference in order to add the domain annex information to the IR nodes
        ir = infer_domain.infer_program(ir, offset_provider=offset_provider)
    offset_provider_type = gtx_common.offset_provider_to_type(offset_provider)
    return dace_backend.build_sdfg_from_gtir(ir, offset_provider_type, column_axis=KDim)


def apply_margin_on_field_domain(
    node: gtir.Expr, dim: gtx_common.Dimension, margin: tuple[int, int]
) -> gtir.Expr:
    """Helper function to narrow the domain in one dimension.

    The `margin` argument specifies two integer offsets, the first to be added
    to the range start, the second to be substracted from the range end.
    """
    domain = domain_utils.SymbolicDomain.from_expr(node)
    domain.ranges[dim] = domain_utils.SymbolicRange(
        im.plus(domain.ranges[dim].start, margin[0]), im.minus(domain.ranges[dim].stop, margin[1])
    )
    return domain.as_expr()


def test_gtir_broadcast():
    val = np.random.rand()
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim])
    testee = gtir.Program(
        id="gtir_broadcast",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
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

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS, skip_domain_inference=True)

    sdfg(a, **FSYMBOLS)
    np.testing.assert_array_equal(a, val)


def test_gtir_cast():
    IFTYPE_FLOAT32 = ts.FieldType(IFTYPE.dims, dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    IFTYPE_BOOL = ts.FieldType(IFTYPE.dims, dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL))
    testee = gtir.Program(
        id="gtir_cast",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE_FLOAT32),
            gtir.Sym(id="z", type=IFTYPE_BOOL),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("eq")(
                    im.cast_as_fieldop("float32")("x"),
                    "y",
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.ones(N, dtype=np.float64) * np.sqrt(2.0)
    b = a.astype(np.float32)
    c = np.empty_like(a, dtype=np.bool_)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    np.testing.assert_array_equal(c, True)


def test_gtir_copy_self():
    testee = gtir.Program(
        id="gtir_copy_self",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=gtir.SymRef(id="x"),
                domain=im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (1, 2)}),
                target=gtir.SymRef(id="x"),
            )
        ],
    )

    a = np.random.rand(N)
    ref = a.copy()

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, **FSYMBOLS)
    assert np.allclose(a, ref)


def test_gtir_tuple_swap():
    testee = gtir.Program(
        id="gtir_tuple_swap",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple("y", "x"),
                domain=im.make_tuple(
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim]),
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                ),
                # TODO(havogt): add a frontend check for this pattern
                target=im.make_tuple("x", "y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = (a.copy(), b.copy())

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(a, ref[1])
    assert np.allclose(b, ref[0])


def test_gtir_tuple_args():
    testee = gtir.Program(
        id="gtir_tuple_args",
        function_definitions=[],
        params=[
            gtir.Sym(
                id="x", type=ts.TupleType(types=[IFTYPE, ts.TupleType(types=[IFTYPE, IFTYPE])])
            ),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus")(
                    im.tuple_get(0, "x"),
                    im.op_as_fieldop("plus")(
                        im.tuple_get(0, im.tuple_get(1, "x")),
                        im.tuple_get(1, im.tuple_get(1, "x")),
                    ),
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    x_fields = (a, a, b)

    tuple_symbols = {
        "__x_0_IDim_range_0": 0,
        "__x_0_IDim_range_1": N,
        "__x_0_stride_0": 1,
        "__x_1_0_IDim_range_0": 0,
        "__x_1_0_IDim_range_1": N,
        "__x_1_0_stride_0": 1,
        "__x_1_1_IDim_range_0": 0,
        "__x_1_1_IDim_range_1": N,
        "__x_1_1_stride_0": 1,
    }

    sdfg(*x_fields, c, **FSYMBOLS, **tuple_symbols)
    assert np.allclose(c, a * 2 + b)


def test_gtir_tuple_expr():
    testee = gtir.Program(
        id="gtir_tuple_expr",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus")(
                    im.tuple_get(0, im.make_tuple("x", im.make_tuple("x", "y"))),
                    im.op_as_fieldop("plus")(
                        im.tuple_get(
                            0, im.tuple_get(1, im.make_tuple("x", im.make_tuple("x", "y")))
                        ),
                        im.tuple_get(
                            1, im.tuple_get(1, im.make_tuple("x", im.make_tuple("x", "y")))
                        ),
                    ),
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, a * 2 + b)


def test_gtir_tuple_broadcast_scalar():
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim])
    testee = gtir.Program(
        id="gtir_tuple_broadcast_scalar",
        function_definitions=[],
        params=[
            gtir.Sym(
                id="x",
                type=ts.TupleType(types=[FLOAT_TYPE, ts.TupleType(types=[FLOAT_TYPE, FLOAT_TYPE])]),
            ),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop("deref", domain)(
                    im.plus(
                        im.tuple_get(0, "x"),
                        im.plus(
                            im.multiplies_(
                                im.tuple_get(
                                    0,
                                    im.tuple_get(1, "x"),
                                ),
                                2.0,
                            ),
                            im.multiplies_(
                                im.tuple_get(
                                    1,
                                    im.tuple_get(1, "x"),
                                ),
                                3.0,
                            ),
                        ),
                    )
                ),
                domain=domain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand()
    d = np.empty(N, dtype=type(a))

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS, skip_domain_inference=True)

    x_fields = (a, b, c)

    sdfg(*x_fields, d, **FSYMBOLS)
    assert np.allclose(d, a + 2 * b + 3 * c)


def test_gtir_zero_dim_fields():
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim])
    empty_domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={})
    testee = gtir.Program(
        id="gtir_zero_dim_fields",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=ts.FieldType(dims=[], dtype=IFTYPE.dtype)),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("multiplies", domain)(
                    "x", im.op_as_fieldop("plus", empty_domain)(1.0, 1.0)
                ),
                domain=domain,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.asarray(np.random.rand())
    b = np.empty(N)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS, skip_domain_inference=True)

    sdfg(a.item(), b, **FSYMBOLS)
    assert np.allclose(b, a * 2)


def test_gtir_tuple_return():
    testee = gtir.Program(
        id="gtir_tuple_return",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(
                id="z", type=ts.TupleType(types=[ts.TupleType(types=[IFTYPE, IFTYPE]), IFTYPE])
            ),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple(im.make_tuple(im.op_as_fieldop("plus")("x", "y"), "x"), "y"),
                domain=im.make_tuple(
                    im.make_tuple(
                        im.get_field_domain(
                            gtx_common.GridType.CARTESIAN,
                            im.tuple_get(0, im.tuple_get(0, "z")),
                            [IDim],
                        ),
                        im.get_field_domain(
                            gtx_common.GridType.CARTESIAN,
                            im.tuple_get(1, im.tuple_get(0, "z")),
                            [IDim],
                        ),
                    ),
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(1, "z"), [IDim]
                    ),
                ),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    z_fields = (np.empty_like(a), np.empty_like(a), np.empty_like(a))

    tuple_symbols = {
        "__z_0_0_IDim_range_0": 0,
        "__z_0_0_IDim_range_1": N,
        "__z_0_0_stride_0": 1,
        "__z_0_1_IDim_range_0": 0,
        "__z_0_1_IDim_range_1": N,
        "__z_0_1_stride_0": 1,
        "__z_1_IDim_range_0": 0,
        "__z_1_IDim_range_1": N,
        "__z_1_stride_0": 1,
    }

    sdfg(a, b, *z_fields, **FSYMBOLS, **tuple_symbols)
    assert np.allclose(z_fields[0], a + b)
    assert np.allclose(z_fields[1], a)
    assert np.allclose(z_fields[2], b)


def test_gtir_tuple_target():
    testee = gtir.Program(
        id="gtir_tuple_target",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.make_tuple(im.op_as_fieldop("plus")("x", 1.0), gtir.SymRef(id="x")),
                domain=im.make_tuple(
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim]),
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                ),
                target=im.make_tuple("x", "y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)
    ref = a.copy()

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(a, ref + 1)
    assert np.allclose(b, ref)


def test_gtir_update():
    stencil1 = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref("a"), im.plus(im.minus(0.0, 2.0), 1.0)))
    )("x")
    stencil2 = im.op_as_fieldop("plus")("x", im.plus(im.minus(0.0, 2.0), 1.0))

    for i, stencil in enumerate([stencil1, stencil2]):
        testee = gtir.Program(
            id=f"gtir_update_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=IFTYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim]),
                    target=gtir.SymRef(id="x"),
                )
            ],
        )
        sdfg = build_dace_sdfg(testee, {})

        a = np.random.rand(N)
        ref = a - 1.0

        sdfg(a, **FSYMBOLS)
        assert np.allclose(a, ref)


def test_gtir_sum2():
    testee = gtir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus")("x", "y"),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    testee = gtir.Program(
        id="sum_2fields_sym",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus")("x", "x"),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, {})

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    stencil1 = im.op_as_fieldop("plus")(
        "x",
        im.op_as_fieldop("plus")("y", "w"),
    )
    stencil2 = im.as_fieldop(
        im.lambda_("a", "b", "c")(im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c"))))
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
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                    target=gtir.SymRef(id="z"),
                )
            ],
        )

        sdfg = build_dace_sdfg(testee, {})

        d = np.empty_like(a)

        sdfg(a, b, c, d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


@pytest.mark.parametrize("s1", [1, 2])
@pytest.mark.parametrize("s2", [1, 2])
def test_gtir_cond(s1, s2):
    testee = gtir.Program(
        id=f"cond_2sums_{s1}_{s2}",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="w", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="s1", type=ts.ScalarType(ts.ScalarKind.INT32)),
            gtir.Sym(id="s2", type=ts.ScalarType(ts.ScalarKind.INT32)),
            gtir.Sym(id="scalar", type=ts.ScalarType(ts.ScalarKind.FLOAT64)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("plus")(
                    "x",
                    im.if_(
                        im.greater("s1", "s2"),
                        im.op_as_fieldop("plus")("y", "scalar"),
                        im.op_as_fieldop("plus")("w", "scalar"),
                    ),
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)
    d = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, d, s1, s2, scalar=1.0, **FSYMBOLS)
    assert np.allclose(d, (a + b + 1) if s1 > s2 else (a + c + 1))


@pytest.mark.xfail(reason="requires function to retrieve the annex tuple domain")
def test_gtir_cond_with_tuple_return():
    testee = gtir.Program(
        id="cond_with_tuple_return",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="w", type=IFTYPE),
            gtir.Sym(id="z", type=ts.TupleType(types=[IFTYPE, IFTYPE])),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.tuple_get(
                    0,
                    im.if_(
                        "pred",
                        im.make_tuple(im.make_tuple("x", "y"), "w"),
                        im.make_tuple(im.make_tuple("y", "x"), "w"),
                    ),
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    tuple_symbols = {
        "__z_0_IDim_range_0": 0,
        "__z_0_IDim_range_1": N,
        "__z_0_stride_0": 1,
        "__z_1_IDim_range_1": 0,
        "__z_1_IDim_range_1": N,
        "__z_1_stride_0": 1,
    }

    for s in [False, True]:
        z_fields = (np.empty_like(a), np.empty_like(a))
        sdfg(a, b, c, *z_fields, pred=np.bool_(s), **FSYMBOLS, **tuple_symbols)
        assert np.allclose(z_fields[0], a if s else b)
        assert np.allclose(z_fields[1], b if s else a)


@pytest.mark.parametrize("s1", [False, True])
@pytest.mark.parametrize("s2", [False, True])
def test_gtir_cond_nested(s1, s2):
    testee = gtir.Program(
        id=f"cond_nested_{int(s1)}_{int(s2)}",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="pred_1", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="pred_2", type=ts.ScalarType(ts.ScalarKind.BOOL)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.if_(
                    "pred_1",
                    im.op_as_fieldop("plus")("x", 1.0),
                    im.if_(
                        "pred_2",
                        im.op_as_fieldop("plus")("x", 2.0),
                        im.op_as_fieldop("plus")("x", 3.0),
                    ),
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, {})

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg(a, b, pred_1=np.bool_(s1), pred_2=np.bool_(s2), **FSYMBOLS)
    assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))


def test_gtir_cartesian_shift_left():
    DELTA = 3.0
    OFFSET = 1

    # cartesian shift with literal integer offset
    stencil1_inlined = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref(im.shift(IDim.value, OFFSET)("a")), DELTA))
    )("x")
    # fieldview flavor of same stencil, in which a temporary field is initialized with the `DELTA` constant value
    stencil1_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a")(im.deref(im.shift(IDim.value, OFFSET)("a"))))("x"),
        im.as_fieldop(im.lambda_()(DELTA))(),
    )

    # use dynamic offset retrieved from field
    stencil2_inlined = im.as_fieldop(
        im.lambda_("a", "off")(im.plus(im.deref(im.shift(IDim.value, im.deref("off"))("a")), DELTA))
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil2_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a", "off")(im.deref(im.shift(IDim.value, im.deref("off"))("a"))))(
            "x", "x_offset"
        ),
        im.as_fieldop(im.lambda_()(DELTA))(),
    )

    # use the result of an arithmetic field operation as dynamic offset
    stencil3_inlined = im.as_fieldop(
        im.lambda_("a", "off")(
            im.plus(im.deref(im.shift(IDim.value, im.plus(im.deref("off"), 0))("a")), DELTA)
        )
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil3_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a", "off")(im.deref(im.shift(IDim.value, im.deref("off"))("a"))))(
            "x",
            im.op_as_fieldop("plus")("x_offset", 0),
        ),
        im.as_fieldop(im.lambda_()(DELTA))(),
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
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=apply_margin_on_field_domain(
                        im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                        IDim,
                        (0, OFFSET),
                    ),
                    target=gtir.SymRef(id="y"),
                )
            ],
        )

        sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

        symbols = FSYMBOLS | {
            "__x_offset_IDim_range_0": 0,
            "__x_offset_IDim_range_1": N,
            "__x_offset_stride_0": 1,
        }

        sdfg(a, a_offset, b, **symbols)
        assert np.allclose(a[OFFSET:] + DELTA, b[:-OFFSET])


def test_gtir_cartesian_shift_right():
    DELTA = 3.0
    OFFSET = 1

    # cartesian shift with literal integer offset
    stencil1_inlined = im.as_fieldop(
        im.lambda_("a")(im.plus(im.deref(im.shift(IDim.value, -OFFSET)("a")), DELTA))
    )("x")
    # fieldview flavor of same stencil, in which a temporary field is initialized with the `DELTA` constant value
    stencil1_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a")(im.deref(im.shift(IDim.value, -OFFSET)("a"))))("x"),
        im.as_fieldop(im.lambda_()(DELTA))(),
    )

    # use dynamic offset retrieved from field
    stencil2_inlined = im.as_fieldop(
        im.lambda_("a", "off")(im.plus(im.deref(im.shift(IDim.value, im.deref("off"))("a")), DELTA))
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil2_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a", "off")(im.deref(im.shift(IDim.value, im.deref("off"))("a"))))(
            "x", "x_offset"
        ),
        im.as_fieldop(im.lambda_()(DELTA))(),
    )

    # use the result of an arithmetic field operation as dynamic offset
    stencil3_inlined = im.as_fieldop(
        im.lambda_("a", "off")(
            im.plus(im.deref(im.shift(IDim.value, im.plus(im.deref("off"), 0))("a")), DELTA)
        )
    )("x", "x_offset")
    # fieldview flavor of same stencil
    stencil3_fieldview = im.op_as_fieldop("plus")(
        im.as_fieldop(im.lambda_("a", "off")(im.deref(im.shift(IDim.value, im.deref("off"))("a"))))(
            "x",
            im.op_as_fieldop("plus")("x_offset", 0),
        ),
        im.as_fieldop(im.lambda_()(DELTA))(),
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
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=apply_margin_on_field_domain(
                        im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                        IDim,
                        (OFFSET, 0),
                    ),
                    target=gtir.SymRef(id="y"),
                )
            ],
        )

        sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

        symbols = FSYMBOLS | {
            "__x_offset_IDim_range_0": 0,
            "__x_offset_IDim_range_1": N,
            "__x_offset_stride_0": 1,
        }

        sdfg(a, a_offset, b, **symbols)
        assert np.allclose(a[:-OFFSET] + DELTA, b[OFFSET:])


def test_gtir_connectivity_shift():
    C2E_neighbor_idx = 2
    E2V_neighbor_idx = 1

    # apply shift 2 times along different dimensions
    stencil1_inlined = im.as_fieldop(
        im.lambda_("it")(
            im.deref(im.shift("C2E", C2E_neighbor_idx)(im.shift("E2V", E2V_neighbor_idx)("it")))
        )
    )("ev_field")

    # fieldview flavor of the same stncil: create an intermediate temporary field
    stencil1_fieldview = im.as_fieldop(
        im.lambda_("it")(im.deref(im.shift("E2V", E2V_neighbor_idx)("it")))
    )(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", C2E_neighbor_idx)("it"))))(
            "ev_field"
        )
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
        )
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
        )
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
        )
    )(
        "ev_field",
        "c2e_offset",
        im.op_as_fieldop("plus")("e2v_offset", 0),
    )

    CE_FTYPE = ts.FieldType(dims=[Cell, Edge], dtype=FLOAT_TYPE)
    EV_FTYPE = ts.FieldType(dims=[Edge, Vertex], dtype=FLOAT_TYPE)
    CELL_OFFSET_FTYPE = ts.FieldType(dims=[Cell], dtype=SIZE_TYPE)
    EDGE_OFFSET_FTYPE = ts.FieldType(dims=[Edge], dtype=SIZE_TYPE)

    connectivity_C2E = SIMPLE_MESH.offset_provider["C2E"]
    connectivity_E2V = SIMPLE_MESH.offset_provider["E2V"]

    ev = np.random.rand(SIMPLE_MESH.num_edges, SIMPLE_MESH.num_vertices)
    ref = ev[connectivity_C2E.asnumpy()[:, C2E_neighbor_idx], :][
        :, connectivity_E2V.asnumpy()[:, E2V_neighbor_idx]
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
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=im.get_field_domain(
                        gtx_common.GridType.UNSTRUCTURED,
                        "ce_field",
                        [Cell, Edge],
                    ),
                    target=gtir.SymRef(id="ce_field"),
                )
            ],
        )

        sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider)

        ce = np.empty([SIMPLE_MESH.num_cells, SIMPLE_MESH.num_edges])

        sdfg(
            ce,
            ev,
            c2e_offset=np.full(SIMPLE_MESH.num_cells, C2E_neighbor_idx, dtype=np.int32),
            e2v_offset=np.full(SIMPLE_MESH.num_edges, E2V_neighbor_idx, dtype=np.int32),
            gt_conn_C2E=connectivity_C2E.ndarray,
            gt_conn_E2V=connectivity_E2V.ndarray,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
            __ce_field_Cell_range_0=0,
            __ce_field_Cell_range_1=SIMPLE_MESH.num_cells,
            __ce_field_Edge_range_0=0,
            __ce_field_Edge_range_1=SIMPLE_MESH.num_edges,
            __ce_field_stride_0=SIMPLE_MESH.num_edges,
            __ce_field_stride_1=1,
            __ev_field_Edge_range_0=0,
            __ev_field_Edge_range_1=SIMPLE_MESH.num_edges,
            __ev_field_Vertex_range_0=0,
            __ev_field_Vertex_range_1=SIMPLE_MESH.num_vertices,
            __ev_field_stride_0=SIMPLE_MESH.num_vertices,
            __ev_field_stride_1=1,
            __c2e_offset_Cell_range_0=0,
            __c2e_offset_Cell_range_1=SIMPLE_MESH.num_cells,
            __c2e_offset_stride_0=1,
            __e2v_offset_Edge_range_0=0,
            __e2v_offset_Edge_range_1=SIMPLE_MESH.num_edges,
            __e2v_offset_stride_0=1,
        )
        assert np.allclose(ce, ref)


def test_gtir_connectivity_shift_chain():
    E2V_neighbor_idx = 1
    V2E_neighbor_idx = 2

    testee = gtir.Program(
        id="connectivity_shift_chain",
        function_definitions=[],
        params=[
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="edges_out", type=EFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(
                    # let domain inference infer the domain here
                    im.lambda_("it")(im.deref(im.shift("E2V", E2V_neighbor_idx)("it"))),
                )(
                    im.as_fieldop(
                        # let domain inference infer the domain here
                        im.lambda_("it")(im.deref(im.shift("V2E", V2E_neighbor_idx)("it"))),
                    )("edges")
                ),
                domain=im.get_field_domain(
                    gtx_common.GridType.UNSTRUCTURED,
                    "edges",
                    [Edge],
                ),
                target=gtir.SymRef(id="edges_out"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider)

    connectivity_E2V = SIMPLE_MESH.offset_provider["E2V"]
    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]

    e = np.random.rand(SIMPLE_MESH.num_edges)
    ref = e[
        connectivity_V2E.asnumpy()[
            connectivity_E2V.asnumpy()[:, E2V_neighbor_idx], V2E_neighbor_idx
        ]
    ]

    # new empty output field
    e_out = np.empty_like(e)

    sdfg(
        e,
        e_out,
        gt_conn_E2V=connectivity_E2V.ndarray,
        gt_conn_V2E=connectivity_V2E.ndarray,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
        __edges_out_Edge_range_0=0,
        __edges_out_Edge_range_1=SIMPLE_MESH.num_edges,
        __edges_out_stride_0=1,
    )
    assert np.allclose(e_out, ref)


def test_gtir_neighbors_as_input():
    MARGIN = 10
    MESH_NUM_LEVELS = 25
    EKFTYPE = ts.FieldType(dims=[Edge, KDim], dtype=FLOAT_TYPE)
    VKFTYPE = ts.FieldType(dims=[Vertex, KDim], dtype=FLOAT_TYPE)
    V2E_KFTYPE = ts.FieldType(dims=[Vertex, V2EDim, KDim], dtype=EFTYPE.dtype)
    gtx_common.check_dims(V2E_KFTYPE.dims)

    init_value = np.random.rand()
    inner_domain = im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "vertices", [Vertex, KDim])
    outer_domain = apply_margin_on_field_domain(inner_domain, KDim, (MARGIN, MARGIN))
    testee = gtir.Program(
        id="neighbors_as_input",
        function_definitions=[],
        params=[
            gtir.Sym(id="v2e_field", type=V2E_KFTYPE),
            gtir.Sym(id="edges", type=EKFTYPE),
            gtir.Sym(id="vertices", type=VKFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "x",
                    im.as_fieldop_neighbors("V2E", "edges", outer_domain),
                )(
                    im.as_fieldop(
                        im.lambda_("it")(
                            im.reduce("plus", im.literal_from_value(init_value))(im.deref("it"))
                        ),
                        inner_domain,
                    )(im.op_as_fieldop(im.map_("divides"), inner_domain)("v2e_field", "x"))
                ),
                domain=outer_domain,
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    # skip domain inference to test correct symbol mapping in let-statements,
    # based on canonical order of field dimensions
    sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider, skip_domain_inference=True)

    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]

    v2e_field = np.random.rand(SIMPLE_MESH.num_vertices, connectivity_V2E.shape[1], MESH_NUM_LEVELS)
    e = np.random.rand(SIMPLE_MESH.num_edges, MESH_NUM_LEVELS)
    v = np.random.rand(SIMPLE_MESH.num_vertices, MESH_NUM_LEVELS)

    v_ref = np.concatenate(
        [
            v[:, :MARGIN],
            list(
                functools.reduce(
                    lambda x, y: x + y,
                    (v2e_values / e[v2e_neighbors])[:, MARGIN:-MARGIN],
                    init_value,
                )
                for v2e_neighbors, v2e_values in zip(
                    connectivity_V2E.asnumpy(), v2e_field, strict=True
                )
            ),
            v[:, -MARGIN:],
        ],
        axis=1,
    )

    symbols = make_mesh_symbols(SIMPLE_MESH) | {
        # override SDFG symbols for array shape and strides because of extra K-dimension
        "__edges_KDim_range_0": 0,
        "__edges_KDim_range_1": e.shape[1],
        "__edges_stride_0": e.strides[0] // e.itemsize,
        "__edges_stride_1": e.strides[1] // e.itemsize,
        "__vertices_KDim_range_0": 0,
        "__vertices_KDim_range_1": v.shape[1],
        "__vertices_stride_0": v.strides[0] // v.itemsize,
        "__vertices_stride_1": v.strides[1] // v.itemsize,
        "__v2e_field_Vertex_range_0": 0,
        "__v2e_field_Vertex_range_1": v2e_field.shape[0],
        "__v2e_field_V2E_range_0": 0,
        "__v2e_field_V2E_range_1": v2e_field.shape[1],
        "__v2e_field_KDim_range_0": 0,
        "__v2e_field_KDim_range_1": v2e_field.shape[2],
        "__v2e_field_stride_0": v2e_field.strides[0] // v2e_field.itemsize,
        "__v2e_field_stride_1": v2e_field.strides[1] // v2e_field.itemsize,
        "__v2e_field_stride_2": v2e_field.strides[2] // v2e_field.itemsize,
    }

    sdfg(v2e_field, e, v, gt_conn_V2E=connectivity_V2E.ndarray, **symbols)
    assert np.allclose(v, v_ref)


def test_gtir_reduce():
    init_value = np.random.rand()
    stencil_inlined = im.as_fieldop(
        im.lambda_("it")(
            im.reduce("plus", im.literal_from_value(init_value))(im.neighbors("V2E", "it"))
        )
    )("edges")
    stencil_fieldview = im.as_fieldop(
        im.lambda_("it")(im.reduce("plus", im.literal_from_value(init_value))(im.deref("it")))
    )(im.as_fieldop_neighbors("V2E", "edges"))

    connectivity_V2E = SIMPLE_MESH.offset_provider["V2E"]

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v_ref = [
        functools.reduce(lambda x, y: x + y, e[v2e_neighbors], init_value)
        for v2e_neighbors in connectivity_V2E.asnumpy()
    ]

    for i, stencil in enumerate([stencil_inlined, stencil_fieldview]):
        testee = gtir.Program(
            id=f"reduce_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="edges", type=EFTYPE),
                gtir.Sym(id="vertices", type=VFTYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=im.get_field_domain(
                        gtx_common.GridType.UNSTRUCTURED,
                        "vertices",
                        [Vertex],
                    ),
                    target=gtir.SymRef(id="vertices"),
                )
            ],
        )
        sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider)

        # new empty output field
        v = np.empty(SIMPLE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            e,
            v,
            gt_conn_V2E=connectivity_V2E.ndarray,
            **FSYMBOLS,
            **make_mesh_symbols(SIMPLE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_with_skip_values():
    init_value = np.random.rand()
    stencil_inlined = im.as_fieldop(
        im.lambda_("it")(
            im.reduce("plus", im.literal_from_value(init_value))(im.neighbors("V2E", "it"))
        )
    )("edges")
    stencil_fieldview = im.as_fieldop(
        im.lambda_("it")(im.reduce("plus", im.literal_from_value(init_value))(im.deref("it")))
    )(im.as_fieldop_neighbors("V2E", "edges"))

    connectivity_V2E = SKIP_VALUE_MESH.offset_provider["V2E"]

    e = np.random.rand(SKIP_VALUE_MESH.num_edges)
    v_ref = [
        functools.reduce(
            lambda x, y: x + y,
            [e[i] if i != gtx_common._DEFAULT_SKIP_VALUE else 0.0 for i in v2e_neighbors],
            init_value,
        )
        for v2e_neighbors in connectivity_V2E.asnumpy()
    ]

    for i, stencil in enumerate([stencil_inlined, stencil_fieldview]):
        testee = gtir.Program(
            id=f"reduce_with_skip_values_{i}",
            function_definitions=[],
            params=[
                gtir.Sym(id="edges", type=EFTYPE),
                gtir.Sym(id="vertices", type=VFTYPE),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=stencil,
                    domain=im.get_field_domain(
                        gtx_common.GridType.UNSTRUCTURED, "vertices", [Vertex]
                    ),
                    target=gtir.SymRef(id="vertices"),
                )
            ],
        )
        sdfg = build_dace_sdfg(testee, SKIP_VALUE_MESH.offset_provider)

        # new empty output field
        v = np.empty(SKIP_VALUE_MESH.num_vertices, dtype=e.dtype)

        sdfg(
            e,
            v,
            gt_conn_V2E=connectivity_V2E.ndarray,
            **FSYMBOLS,
            **make_mesh_symbols(SKIP_VALUE_MESH),
        )
        assert np.allclose(v, v_ref)


def test_gtir_reduce_dot_product():
    init_value = np.random.rand()

    connectivity_V2E = SKIP_VALUE_MESH.offset_provider["V2E"]

    v2e_field = np.random.rand(*connectivity_V2E.shape)
    e = np.random.rand(SKIP_VALUE_MESH.num_edges)
    v = np.empty(SKIP_VALUE_MESH.num_vertices, dtype=e.dtype)
    v_ref = [
        functools.reduce(
            lambda x, y: x + y,
            map(
                lambda x: 0.0 if x[1] == gtx_common._DEFAULT_SKIP_VALUE else x[0],
                zip((e[v2e_neighbors] * v2e_values) + 1.0, v2e_neighbors),
            ),
            init_value,
        )
        for v2e_neighbors, v2e_values in zip(connectivity_V2E.asnumpy(), v2e_field)
    ]

    testee = gtir.Program(
        id=f"reduce_dot_product",
        function_definitions=[],
        params=[
            gtir.Sym(id="v2e_field", type=V2E_FTYPE),
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(
                        im.reduce("plus", im.literal_from_value(init_value))(im.deref("it"))
                    )
                )(
                    im.op_as_fieldop(im.map_("plus"))(
                        im.op_as_fieldop(im.map_("multiplies"))(
                            im.as_fieldop_neighbors("V2E", "edges"),
                            "v2e_field",
                        ),
                        im.op_as_fieldop("make_const_list")(1.0),
                    )
                ),
                domain=im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "vertices", [Vertex]),
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, SKIP_VALUE_MESH.offset_provider)

    sdfg(
        v2e_field,
        e,
        v,
        gt_conn_V2E=connectivity_V2E.ndarray,
        **make_mesh_symbols(SKIP_VALUE_MESH),
        __v2e_field_Vertex_range_0=0,
        __v2e_field_Vertex_range_1=SKIP_VALUE_MESH.num_vertices,
        __v2e_field_stride_0=connectivity_V2E.shape[1],
        __v2e_field_stride_1=1,
    )
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_gtir_reduce_with_cond_neighbors(use_sparse):
    init_value = np.random.rand()
    testee = gtir.Program(
        id=f"reduce_with_cond_neighbors_{int(use_sparse)}",
        function_definitions=[],
        params=[
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
            gtir.Sym(id="v2e_field", type=V2E_FTYPE),
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(
                        im.reduce("plus", im.literal_from_value(init_value))(im.deref("it"))
                    )
                )(
                    im.if_(
                        "pred",
                        "v2e_field",
                        im.as_fieldop_neighbors("V2E", "edges"),
                    )
                ),
                domain=im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "vertices", [Vertex]),
                target=gtir.SymRef(id="vertices"),
            )
        ],
    )

    connectivity_V2E = SKIP_VALUE_MESH.offset_provider["V2E"]

    sdfg = build_dace_sdfg(testee, SKIP_VALUE_MESH.offset_provider)

    v2e_field = np.random.rand(*connectivity_V2E.shape)
    e = np.random.rand(SKIP_VALUE_MESH.num_edges)
    v = np.empty(SKIP_VALUE_MESH.num_vertices, dtype=e.dtype)
    v_ref = [
        functools.reduce(
            lambda x, y: x + y,
            [
                v if i != gtx_common._DEFAULT_SKIP_VALUE else 0.0
                for i, v in zip(v2e_neighbors, v2e_values, strict=True)
            ],
            init_value,
        )
        if use_sparse
        else functools.reduce(
            lambda x, y: x + y,
            [e[i] if i != gtx_common._DEFAULT_SKIP_VALUE else 0.0 for i in v2e_neighbors],
            init_value,
        )
        for v2e_neighbors, v2e_values in zip(connectivity_V2E.asnumpy(), v2e_field, strict=True)
    ]
    sdfg(
        np.bool_(use_sparse),
        v2e_field,
        e,
        v,
        gt_conn_V2E=connectivity_V2E.ndarray,
        **FSYMBOLS,
        **make_mesh_symbols(SKIP_VALUE_MESH),
        __v2e_field_Vertex_range_0=0,
        __v2e_field_Vertex_range_1=SKIP_VALUE_MESH.num_vertices,
        __v2e_field_stride_0=connectivity_V2E.shape[1],
        __v2e_field_stride_1=1,
    )
    assert np.allclose(v, v_ref)


def test_gtir_symbolic_domain():
    MARGIN = 2
    assert MARGIN < N
    OFFSET = 1000 * 1000 * 1000
    shift_left_stencil = im.lambda_("a")(im.deref(im.shift(IDim.value, OFFSET)("a")))
    shift_right_stencil = im.lambda_("a")(im.deref(im.shift(IDim.value, -OFFSET)("a")))
    testee = gtir.Program(
        id="symbolic_domain",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="size", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "xᐞ1",
                    im.op_as_fieldop("multiplies")(
                        4.0,
                        im.as_fieldop(shift_left_stencil)("x"),
                    ),
                )(
                    im.let(
                        "xᐞ2",
                        im.op_as_fieldop("multiplies")(
                            3.0,
                            im.as_fieldop(shift_right_stencil)("x"),
                        ),
                    )(
                        im.let(
                            "xᐞ3",
                            im.as_fieldop(shift_right_stencil)("xᐞ1"),
                        )(
                            im.let(
                                "xᐞ4",
                                im.as_fieldop(shift_left_stencil)("xᐞ2"),
                            )(
                                im.let("xᐞ5", im.op_as_fieldop("plus")("xᐞ3", "xᐞ4"))(
                                    im.op_as_fieldop("plus")("xᐞ5", "x")
                                )
                            )
                        )
                    )
                ),
                domain=apply_margin_on_field_domain(
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                    IDim,
                    (MARGIN, MARGIN),
                ),
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = np.concatenate((b[0:MARGIN], a[MARGIN : N - MARGIN] * 8, b[N - MARGIN : N]))

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, size=N, **FSYMBOLS)
    assert np.allclose(b, ref)


def test_gtir_let_lambda():
    testee = gtir.Program(
        id="let_lambda",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                # `xᐞ1` is a let-lambda expression representing `x * 3`
                # `xᐞ2` is a let-lambda expression representing `x * 4`
                #  - note that the let-symbol `xᐞ2` is used twice, in a nested let-expression, to test aliasing of the symbol
                # `xᐞ3` is a let-lambda expression simply accessing `x` field symref
                expr=im.let("xᐞ1", im.op_as_fieldop("multiplies")(3.0, "x"))(
                    im.let(
                        "xᐞ2",
                        im.let("xᐞ2", im.op_as_fieldop("multiplies")(2.0, "x"))(
                            im.op_as_fieldop("plus")("xᐞ2", "xᐞ2")
                        ),
                    )(
                        im.let("xᐞ3", "x")(
                            im.op_as_fieldop("plus")("xᐞ1", im.op_as_fieldop("plus")("xᐞ2", "xᐞ3"))
                        )
                    )
                ),
                domain=apply_margin_on_field_domain(
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]), IDim, (1, 1)
                ),
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = np.concatenate((b[0:1], a[1 : N - 1] * 8, b[N - 1 : N]))

    sdfg = build_dace_sdfg(testee, {})

    sdfg(a, b, **FSYMBOLS)
    assert np.allclose(b, ref)


def test_gtir_let_lambda_scalar_expression():
    domain_inner = im.domain(gtx_common.GridType.CARTESIAN, ranges={IDim: (1, "size_inner")})
    domain_outer = im.get_field_domain(
        gtx_common.GridType.CARTESIAN,
        "y",
        [IDim],
    )
    testee = gtir.Program(
        id="let_lambda_scalar_expression",
        function_definitions=[],
        params=[
            gtir.Sym(id="a", type=IFTYPE.dtype),
            gtir.Sym(id="b", type=IFTYPE.dtype),
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                # we create an inner symbol that will be mapped to a scalar expression of the parent node
                expr=im.let(
                    "size_inner",
                    im.plus(
                        domain_utils.SymbolicDomain.from_expr(domain_outer).ranges[IDim].stop, 1
                    ),
                )(
                    im.let("tmp", im.multiplies_("a", "b"))(
                        im.as_fieldop(
                            im.lambda_("a")(im.deref(im.shift(IDim.value, 1)("a"))), domain_outer
                        )(
                            im.op_as_fieldop("multiplies", domain_inner)(
                                "x", im.multiplies_("tmp", "tmp")
                            )
                        )
                    )
                ),
                domain=domain_outer,
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand()
    b = np.random.rand()
    c = np.random.rand(N + 1)
    d = np.zeros(N)

    # We use `skip_domain_inference=True` to avoid propagating the compute domain
    # to the inner expression, so that the mapping of the scalar expression `size + 1`
    # to the symbol `inner_size` is preserved, for which we want to test the lowering.
    sdfg = build_dace_sdfg(testee, offset_provider=CARTESIAN_OFFSETS, skip_domain_inference=True)

    sdfg(a, b, c, d, **(FSYMBOLS | {"__x_0_range_1": N + 1}))
    assert np.allclose(d, (a * a * b * b * c[1 : N + 1]))


def test_gtir_let_lambda_with_connectivity():
    C2E_neighbor_idx = 1
    C2V_neighbor_idx = 2

    connectivity_C2E = SIMPLE_MESH.offset_provider["C2E"]
    connectivity_C2V = SIMPLE_MESH.offset_provider["C2V"]

    testee = gtir.Program(
        id="let_lambda_with_connectivity",
        function_definitions=[],
        params=[
            gtir.Sym(id="cells", type=CFTYPE),
            gtir.Sym(id="edges", type=EFTYPE),
            gtir.Sym(id="vertices", type=VFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "x1",
                    im.as_fieldop(
                        im.lambda_("it")(im.deref(im.shift("C2E", C2E_neighbor_idx)("it")))
                    )("edges"),
                )(
                    im.let(
                        "x2",
                        im.as_fieldop(
                            im.lambda_("it")(im.deref(im.shift("C2V", C2V_neighbor_idx)("it")))
                        )("vertices"),
                    )(im.op_as_fieldop("plus")("x1", "x2"))
                ),
                domain=im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "cells", [Cell]),
                target=gtir.SymRef(id="cells"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider)

    e = np.random.rand(SIMPLE_MESH.num_edges)
    v = np.random.rand(SIMPLE_MESH.num_vertices)
    c = np.empty(SIMPLE_MESH.num_cells)
    ref = (
        e[connectivity_C2E.asnumpy()[:, C2E_neighbor_idx]]
        + v[connectivity_C2V.asnumpy()[:, C2V_neighbor_idx]]
    )

    sdfg(
        cells=c,
        edges=e,
        vertices=v,
        gt_conn_C2E=connectivity_C2E.ndarray,
        gt_conn_C2V=connectivity_C2V.ndarray,
        **FSYMBOLS,
        **make_mesh_symbols(SIMPLE_MESH),
    )
    assert np.allclose(c, ref)


def test_gtir_let_lambda_with_origin():
    MESH_NUM_LEVELS = 25
    C2E_neighbor_idx = 1

    CKFTYPE = ts.FieldType(dims=[Cell, KDim], dtype=FLOAT_TYPE)
    EKFTYPE = ts.FieldType(dims=[Edge, KDim], dtype=FLOAT_TYPE)

    testee = gtir.Program(
        id="let_lambda_with_origin",
        function_definitions=[],
        params=[
            gtir.Sym(id="cells", type=CKFTYPE),
            gtir.Sym(id="edges", type=EKFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("e1", im.op_as_fieldop("plus")("edges", 1.0))(
                    im.as_fieldop(
                        im.lambda_("it")(im.deref(im.shift("C2E", C2E_neighbor_idx)("it"))),
                    )("e1")
                ),
                domain=apply_margin_on_field_domain(
                    im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "cells", [Cell, KDim]),
                    KDim,
                    (1, 0),
                ),
                target=gtir.SymRef(id="cells"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, SIMPLE_MESH.offset_provider)

    c = np.random.rand(SIMPLE_MESH.num_cells, MESH_NUM_LEVELS)
    e = np.random.rand(SIMPLE_MESH.num_edges, MESH_NUM_LEVELS)
    connectivity_C2E = SIMPLE_MESH.offset_provider["C2E"]
    ref = np.concatenate(
        (c[:, :1], e[connectivity_C2E.asnumpy()[:, C2E_neighbor_idx], 1:] + 1.0), axis=1
    )

    symbols = make_mesh_symbols(SIMPLE_MESH) | {
        "__cells_KDim_range_0": 0,
        "__cells_KDim_range_1": MESH_NUM_LEVELS,
        "__cells_stride_0": c.strides[0] // c.itemsize,
        "__cells_stride_1": c.strides[1] // c.itemsize,
        "__edges_KDim_range_0": 0,
        "__edges_KDim_range_1": MESH_NUM_LEVELS,
        "__edges_stride_0": e.strides[0] // e.itemsize,
        "__edges_stride_1": e.strides[1] // e.itemsize,
    }

    sdfg(
        cells=c,
        edges=e,
        gt_conn_C2E=connectivity_C2E.ndarray,
        **symbols,
    )

    assert np.allclose(c, ref)


@pytest.mark.parametrize("s", [False, True])
def test_gtir_let_lambda_with_cond(s):
    testee = gtir.Program(
        id=f"let_lambda_with_cond_{int(s)}",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("x1", "x")(
                    im.let("x2", im.op_as_fieldop("multiplies")(2.0, "x"))(
                        im.if_("pred", "x1", "x2")
                    )
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", [IDim]),
                target=gtir.SymRef(id="y"),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, {})

    a = np.random.rand(N)
    b = np.empty_like(a)
    sdfg(a, b, pred=np.bool_(s), **FSYMBOLS)
    assert np.allclose(b, a if s else a * 2)


def test_gtir_let_lambda_with_tuple1():
    inner_domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim])
    testee = gtir.Program(
        id="let_lambda_with_tuple1",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=ts.TupleType(types=[IFTYPE, IFTYPE])),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let(
                    "t",
                    im.make_tuple(
                        im.make_tuple(im.op_as_fieldop("plus", inner_domain)("x", "y"), "x"), "y"
                    ),
                )(im.make_tuple(im.tuple_get(1, im.tuple_get(0, "t")), im.tuple_get(1, "t"))),
                domain=im.make_tuple(
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(0, "z"), [IDim]
                    ),
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(1, "z"), [IDim]
                    ),
                ),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    # TODO(edopao): remove `skip_domain_inference=True` once this error is fixed
    #   in domain inference: 'target_domain' cannot be 'NEVER' unless `allow_uninferred=True`
    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS, skip_domain_inference=True)

    z_fields = (np.zeros_like(a), np.zeros_like(a))
    a_ref = np.concatenate((z_fields[0][:1], a[1 : N - 1], z_fields[0][N - 1 :]))
    b_ref = np.concatenate((z_fields[1][:1], b[1 : N - 1], z_fields[1][N - 1 :]))

    tuple_symbols = {
        "__z_0_IDim_range_0": 1,
        "__z_0_IDim_range_1": N - 1,
        "__z_0_stride_0": 1,
        "__z_1_IDim_range_0": 1,
        "__z_1_IDim_range_1": N - 1,
        "__z_1_stride_0": 1,
    }

    sdfg(a, b, z_fields[0][1 : N - 1], z_fields[1][1 : N - 1], **FSYMBOLS, **tuple_symbols)
    assert np.allclose(z_fields[0], a_ref)
    assert np.allclose(z_fields[1], b_ref)


def test_gtir_let_lambda_with_tuple2():
    inner_domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim])
    val = np.random.rand()
    testee = gtir.Program(
        id="let_lambda_with_tuple2",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=ts.TupleType(types=[IFTYPE, IFTYPE, IFTYPE])),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("s", im.as_fieldop("deref", inner_domain)(val))(
                    im.let("t", im.make_tuple("x", "y"))(
                        im.let("p", im.op_as_fieldop("plus", inner_domain)("x", "y"))(
                            im.make_tuple("p", "s", im.tuple_get(1, "t"))
                        )
                    )
                ),
                domain=im.make_tuple(
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(0, "z"), [IDim]
                    ),
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(1, "z"), [IDim]
                    ),
                    im.get_field_domain(
                        gtx_common.GridType.CARTESIAN, im.tuple_get(2, "z"), [IDim]
                    ),
                ),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    z_fields = (np.empty_like(a), np.empty_like(a), np.empty_like(a))

    tuple_symbols = {
        "__z_0_IDim_range_0": 0,
        "__z_0_IDim_range_1": N,
        "__z_0_stride_0": 1,
        "__z_1_IDim_range_0": 0,
        "__z_1_IDim_range_1": N,
        "__z_1_stride_0": 1,
        "__z_2_IDim_range_0": 0,
        "__z_2_IDim_range_1": N,
        "__z_2_stride_0": 1,
    }

    sdfg(a, b, *z_fields, **FSYMBOLS, **tuple_symbols)
    assert np.allclose(z_fields[0], a + b)
    assert np.allclose(z_fields[1], val)
    assert np.allclose(z_fields[2], b)


@pytest.mark.parametrize("s", [False, True])
def test_gtir_if_scalars(s):
    testee = gtir.Program(
        id=f"if_scalars_{int(s)}",
        function_definitions=[],
        params=[
            gtir.Sym(
                id="x",
                type=ts.TupleType(types=[IFTYPE, ts.TupleType(types=[SIZE_TYPE, SIZE_TYPE])]),
            ),
            gtir.Sym(id="z", type=IFTYPE),
            gtir.Sym(id="pred", type=ts.ScalarType(ts.ScalarKind.BOOL)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("f", im.tuple_get(0, "x"))(
                    im.let("g", im.tuple_get(1, "x"))(
                        im.let("y_0", im.tuple_get(0, "g"))(
                            im.let("y_1", im.tuple_get(1, "g"))(
                                im.op_as_fieldop("plus")(
                                    "f",
                                    im.if_(
                                        "pred",
                                        im.cast_("y_0", "float64"),
                                        im.cast_("y_1", "float64"),
                                    ),
                                )
                            )
                        )
                    )
                ),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)
    d1 = np.random.randint(0, 1000)
    d2 = np.random.randint(0, 1000)

    sdfg = build_dace_sdfg(testee, {})

    tuple_symbols = {
        "__x_0_IDim_range_0": 0,
        "__x_0_IDim_range_1": N,
        "__x_0_stride_0": 1,
        "__x_1_0_IDim_range_0": 0,
        "__x_1_0_IDim_range_1": N,
        "__x_1_0_stride_0": 1,
        "__x_1_1_IDim_range_0": 0,
        "__x_1_1_IDim_range_1": N,
        "__x_1_1_stride_0": 1,
    }

    sdfg(x_0=a, x_1_0=d1, x_1_1=d2, z=b, pred=np.bool_(s), **FSYMBOLS, **tuple_symbols)
    assert np.allclose(b, (a + d1 if s else a + d2))


def test_gtir_if_values():
    testee = gtir.Program(
        id="if_values",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=IFTYPE),
            gtir.Sym(id="y", type=IFTYPE),
            gtir.Sym(id="z", type=IFTYPE),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.op_as_fieldop("if_")(im.op_as_fieldop("less")("x", "y"), "x", "y"),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, **FSYMBOLS)
    assert np.allclose(c, np.where(a < b, a, b))


def test_gtir_index():
    MARGIN = 2
    assert (MARGIN * 2) < N

    testee = gtir.Program(
        id="gtir_index",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=ts.FieldType(dims=[IDim], dtype=SIZE_TYPE)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.let("i", im.index(IDim))(
                    im.op_as_fieldop("plus")(
                        "i",
                        im.as_fieldop(im.lambda_("a")(im.deref(im.shift(IDim.value, 1)("a"))))("i"),
                    )
                ),
                domain=apply_margin_on_field_domain(
                    im.get_field_domain(gtx_common.GridType.CARTESIAN, "x", [IDim]),
                    IDim,
                    (MARGIN, MARGIN),
                ),
                target=gtir.SymRef(id="x"),
            )
        ],
    )

    v = np.zeros(N, dtype=np.int32)

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    ref = np.concatenate(
        (v[:MARGIN], np.arange(MARGIN, N - MARGIN, dtype=np.int32) * 2 + 1, v[N - MARGIN :])
    )

    sdfg(v, **FSYMBOLS)
    assert np.all(v == ref)


def test_gtir_concat_where():
    SUBSET_SIZE = 5
    assert SUBSET_SIZE < N
    domain_cond_lhs = im.domain(
        gtx_common.GridType.CARTESIAN, {IDim: (gtir.InfinityLiteral.NEGATIVE, N - SUBSET_SIZE)}
    )
    domain_cond_rhs = im.domain(
        gtx_common.GridType.CARTESIAN, {IDim: (SUBSET_SIZE, gtir.InfinityLiteral.POSITIVE)}
    )

    concat_expr_lhs = im.concat_where(
        domain_cond_lhs,
        im.as_fieldop("deref")("x"),
        im.as_fieldop("deref")("y"),
    )
    concat_expr_rhs = im.concat_where(
        domain_cond_rhs,
        im.as_fieldop("deref")("y"),
        im.as_fieldop("deref")("x"),
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    ref = np.concatenate((a[:SUBSET_SIZE], b[SUBSET_SIZE:]))

    for concat_expr, suffix in [(concat_expr_lhs, "lhs"), (concat_expr_rhs, "rhs")]:
        testee = gtir.Program(
            id=f"gtir_concat_where_{suffix}",
            function_definitions=[],
            params=[
                gtir.Sym(id="x", type=ts.FieldType(dims=[IDim], dtype=SIZE_TYPE)),
                gtir.Sym(id="y", type=ts.FieldType(dims=[IDim], dtype=SIZE_TYPE)),
                gtir.Sym(id="z", type=ts.FieldType(dims=[IDim], dtype=SIZE_TYPE)),
            ],
            declarations=[],
            body=[
                gtir.SetAt(
                    expr=concat_expr,
                    domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim]),
                    target=gtir.SymRef(id="z"),
                )
            ],
        )

        # run domain inference in order to add the domain annex information to the concat_where node.
        testee = infer_domain.infer_program(testee, offset_provider=CARTESIAN_OFFSETS)
        sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)
        c = np.empty_like(a)

        sdfg(a, b, c, **FSYMBOLS)
        np.allclose(c, ref)


def test_gtir_concat_where_two_dimensions():
    M, N = (30, 20)
    domain = im.domain(gtx_common.GridType.CARTESIAN, {IDim: (0, 30), JDim: (0, 20)})
    domain_cond1 = im.domain(
        gtx_common.GridType.CARTESIAN, {JDim: (10, gtir.InfinityLiteral.POSITIVE)}
    )
    domain_cond2 = im.domain(
        gtx_common.GridType.CARTESIAN, {IDim: (gtir.InfinityLiteral.NEGATIVE, 20)}
    )

    testee = gtir.Program(
        id=f"gtir_concat_where_two_dimensions",
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=ts.FieldType(dims=[IDim, JDim], dtype=FLOAT_TYPE)),
            gtir.Sym(id="y", type=ts.FieldType(dims=[IDim, JDim], dtype=FLOAT_TYPE)),
            gtir.Sym(id="w", type=ts.FieldType(dims=[IDim, JDim], dtype=FLOAT_TYPE)),
            gtir.Sym(id="z", type=ts.FieldType(dims=[IDim, JDim], dtype=FLOAT_TYPE)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.concat_where(
                    domain_cond1,  # 0, 30; 10,20
                    im.concat_where(
                        domain_cond2,
                        im.as_fieldop("deref")("x"),
                        im.as_fieldop("deref")("y"),
                    ),
                    im.as_fieldop("deref")("w"),
                ),
                domain=domain,
                target=gtir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(M, N)
    b = np.random.rand(M, N)
    c = np.random.rand(M, N)
    d = np.empty_like(a)
    ref = np.concatenate(
        (c[:, :10], np.concatenate((a[:20, :], b[20:, :]), axis=0)[:, 10:]), axis=1
    )

    field_symbols = {
        "__x_IDim_range_0": 0,
        "__x_IDim_range_1": a.shape[0],
        "__x_JDim_range_0": 0,
        "__x_JDim_range_1": a.shape[1],
        "__x_stride_0": a.strides[0] // a.itemsize,
        "__x_stride_1": a.strides[1] // a.itemsize,
        "__y_IDim_range_0": 0,
        "__y_IDim_range_1": b.shape[0],
        "__y_JDim_range_0": 0,
        "__y_JDim_range_1": b.shape[1],
        "__y_stride_0": b.strides[0] // b.itemsize,
        "__y_stride_1": b.strides[1] // b.itemsize,
        "__w_IDim_range_0": 0,
        "__w_IDim_range_1": c.shape[0],
        "__w_JDim_range_0": 0,
        "__w_JDim_range_1": c.shape[1],
        "__w_stride_0": c.strides[0] // c.itemsize,
        "__w_stride_1": c.strides[1] // c.itemsize,
        "__z_IDim_range_0": 0,
        "__z_IDim_range_1": d.shape[0],
        "__z_JDim_range_0": 0,
        "__z_JDim_range_1": d.shape[1],
        "__z_stride_0": d.strides[0] // d.itemsize,
        "__z_stride_1": d.strides[1] // d.itemsize,
    }

    # run domain inference in order to add the domain annex information to the concat_where node.
    testee = infer_domain.infer_program(testee, offset_provider=CARTESIAN_OFFSETS)
    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    sdfg(a, b, c, d, **field_symbols)

    np.allclose(d, ref)


@pytest.mark.parametrize(
    ["id", "use_symbolic_column_size"],
    [("gtir_scan_with_constant_column_size", False), ("gtir_scan_with_symbolic_column_size", True)],
    ids=["constant_column_size", "symbolic_column_size"],
)
def test_gtir_scan(id, use_symbolic_column_size):
    K = 20
    VAL = 1.2
    domain = im.get_field_domain(gtx_common.GridType.CARTESIAN, "z", [IDim, KDim])
    if use_symbolic_column_size:
        sd = domain_utils.SymbolicDomain.from_expr(domain)
        sd.ranges[KDim] = domain_utils.SymbolicRange(
            im.literal_from_value(0), im.literal_from_value(K)
        )
        domain = sd.as_expr()
    testee = gtir.Program(
        id=id,
        function_definitions=[],
        params=[
            gtir.Sym(id="x", type=ts.FieldType(dims=[IDim, KDim], dtype=FLOAT_TYPE)),
            gtir.Sym(id="y", type=ts.FieldType(dims=[IDim, KDim], dtype=FLOAT_TYPE)),
            gtir.Sym(id="z", type=ts.FieldType(dims=[IDim, KDim], dtype=BOOL_TYPE)),
        ],
        declarations=[],
        body=[
            gtir.SetAt(
                expr=im.as_fieldop(
                    im.scan(
                        im.lambda_("state", "inp")(
                            im.if_(
                                im.tuple_get(1, "state"),
                                im.make_tuple(
                                    im.plus(VAL, im.deref("inp")),
                                    False,
                                ),
                                im.make_tuple(
                                    im.plus(im.tuple_get(0, "state"), im.deref("inp")),
                                    False,
                                ),
                            )
                        ),
                        True,
                        im.make_tuple(0.0, True),
                    )
                )("x"),
                domain=im.make_tuple(domain, domain),
                target=im.make_tuple(gtir.SymRef(id="y"), gtir.SymRef(id="z")),
            )
        ],
    )

    sdfg = build_dace_sdfg(testee, CARTESIAN_OFFSETS)

    a = np.random.rand(N, K)
    b = np.random.rand(N, K)
    z = np.full([N, K], False, dtype=np.bool_)
    ref = np.add.accumulate(a, axis=1) + VAL

    symbols = FSYMBOLS | {
        "__x_KDim_range_0": 0,
        "__x_KDim_range_1": a.shape[1],
        "__x_stride_0": a.strides[0] // a.itemsize,
        "__x_stride_1": a.strides[1] // a.itemsize,
        "__y_KDim_range_0": 0,
        "__y_KDim_range_1": b.shape[1],
        "__y_stride_0": b.strides[0] // b.itemsize,
        "__y_stride_1": b.strides[1] // b.itemsize,
        "__z_KDim_range_0": 0,
        "__z_KDim_range_1": z.shape[1],
        "__z_stride_0": z.strides[0] // z.itemsize,
        "__z_stride_1": z.strides[1] // z.itemsize,
    }

    sdfg(a, b, z, **symbols)
    assert np.allclose(b, ref)
