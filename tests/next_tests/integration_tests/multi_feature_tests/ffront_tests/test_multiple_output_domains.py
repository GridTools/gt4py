# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import gt4py.next as gtx
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    JDim,
    KDim,
    C2E,
    E2V,
    V2E,
    Edge,
    EField,
    CField,
    VField,
    Cell,
    Vertex,
    cartesian_case,
    Case,
    IField,
    JField,
    KField,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)

from gt4py.next import common

KHalfDim = gtx.Dimension("KHalf", kind=gtx.DimensionKind.VERTICAL)
pytestmark = pytest.mark.uses_cartesian_shift


@gtx.field_operator
def testee_orig(a: IField, b: IField) -> tuple[IField, IField]:
    return b, a


@gtx.program
def prog_orig(
    a: IField,
    b: IField,
    out_a: IField,
    out_b: IField,
    i_size: gtx.int32,
):
    testee_orig(a, b, out=(out_b, out_a), domain={IDim: (0, i_size)})


def test_program_orig(cartesian_case):
    a = cases.allocate(cartesian_case, prog_orig, "a")()
    b = cases.allocate(cartesian_case, prog_orig, "b")()
    out_a = cases.allocate(cartesian_case, prog_orig, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_orig, "out_b")()

    cases.verify(
        cartesian_case,
        prog_orig,
        a,
        b,
        out_a,
        out_b,
        cartesian_case.default_sizes[IDim],
        inout=(out_b, out_a),
        ref=(b, a),
    )


@gtx.program
def prog_no_domain(
    a: IField,
    b: IField,
    out_a: IField,
    out_b: IField,
):
    testee_orig(a, b, out=(out_b, out_a))


def test_program_no_domain(cartesian_case):
    a = cases.allocate(cartesian_case, prog_no_domain, "a")()
    b = cases.allocate(cartesian_case, prog_no_domain, "b")()
    out_a = cases.allocate(cartesian_case, prog_no_domain, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_no_domain, "out_b")()

    cases.verify(
        cartesian_case,
        prog_no_domain,
        a,
        b,
        out_a,
        out_b,
        inout=(out_b, out_a),
        ref=(b, a),
    )


@gtx.field_operator
def testee(a: IField, b: JField) -> tuple[JField, IField]:
    return b, a


@gtx.program
def prog(
    a: IField,
    b: JField,
    out_a: IField,
    out_b: JField,
    i_size: gtx.int32,
    j_size: gtx.int32,
):
    testee(a, b, out=(out_b, out_a), domain=({JDim: (0, j_size)}, {IDim: (0, i_size)}))


def test_program(cartesian_case):
    a = cases.allocate(cartesian_case, prog, "a")()
    b = cases.allocate(cartesian_case, prog, "b")()
    out_a = cases.allocate(cartesian_case, prog, "out_a")()
    out_b = cases.allocate(cartesian_case, prog, "out_b")()

    cases.verify(
        cartesian_case,
        prog,
        a,
        b,
        out_a,
        out_b,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        inout=(out_b, out_a),
        ref=(b, a),
    )


@gtx.program
def prog_out_as_tuple(
    a: IField,
    b: JField,
    out: tuple[JField, IField],
    i_size: gtx.int32,
    j_size: gtx.int32,
):
    testee(a, b, out=out, domain=({JDim: (0, j_size)}, {IDim: (0, i_size)}))


def test_program_out_as_tuple(
    cartesian_case,
):  # TODO: this fails for most backends, merge PR #1893 first
    a = cases.allocate(cartesian_case, prog_out_as_tuple, "a")()
    b = cases.allocate(cartesian_case, prog_out_as_tuple, "b")()
    out = cases.allocate(cartesian_case, prog_out_as_tuple, "out")()

    cases.verify(
        cartesian_case,
        prog_out_as_tuple,
        a,
        b,
        out,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        inout=(out),
        ref=(b, a),
    )


@gtx.field_operator
def testee_nested_tuples(
    a: IField,
    b: JField,
    c: KField,
) -> tuple[
    tuple[IField, JField],
    KField,
]:
    return (a, b), c


@gtx.program
def prog_nested_tuples(
    a: IField,
    b: JField,
    c: KField,
    out_a: IField,
    out_b: JField,
    out_c: KField,
    i_size: gtx.int32,
    j_size: gtx.int32,
    k_size: gtx.int32,
):
    testee_nested_tuples(
        a,
        b,
        c,
        out=((out_a, out_b), out_c),
        domain=(({IDim: (0, i_size)}, {JDim: (0, j_size)}), {KDim: (0, k_size)}),
    )


def test_program_nested_tuples(
    cartesian_case,
):
    a = cases.allocate(cartesian_case, prog_nested_tuples, "a")()
    b = cases.allocate(cartesian_case, prog_nested_tuples, "b")()
    c = cases.allocate(cartesian_case, prog_nested_tuples, "c")()
    out_a = cases.allocate(cartesian_case, prog_nested_tuples, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_nested_tuples, "out_b")()
    out_c = cases.allocate(cartesian_case, prog_nested_tuples, "out_c")()

    cases.verify(
        cartesian_case,
        prog_nested_tuples,
        a,
        b,
        c,
        out_a,
        out_b,
        out_c,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        cartesian_case.default_sizes[KDim],
        inout=((out_a, out_b), out_c),
        ref=((a, b), c),
    )


@gtx.field_operator
def testee_double_nested_tuples(
    a: IField,
    b: JField,
    c: KField,
) -> tuple[
    tuple[
        IField,
        tuple[JField, KField],
    ],
    KField,
]:
    return (a, (b, c)), c


@gtx.program
def prog_double_nested_tuples(
    a: IField,
    b: JField,
    c: KField,
    out_a: IField,
    out_b: JField,
    out_c: KField,
    i_size: gtx.int32,
    j_size: gtx.int32,
    k_size: gtx.int32,
):
    testee_double_nested_tuples(
        a,
        b,
        c,
        out=((out_a, (out_b, out_c)), out_c),
        domain=(
            ({IDim: (0, i_size)}, ({JDim: (0, j_size)}, {KDim: (0, k_size)})),
            {KDim: (0, k_size)},
        ),
    )


def test_program_double_nested_tuples(
    cartesian_case,
):
    a = cases.allocate(cartesian_case, prog_double_nested_tuples, "a")()
    b = cases.allocate(cartesian_case, prog_double_nested_tuples, "b")()
    c = cases.allocate(cartesian_case, prog_double_nested_tuples, "c")()
    out_a = cases.allocate(cartesian_case, prog_double_nested_tuples, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_double_nested_tuples, "out_b")()
    out_c = cases.allocate(cartesian_case, prog_double_nested_tuples, "out_c")()

    cases.verify(
        cartesian_case,
        prog_double_nested_tuples,
        a,
        b,
        c,
        out_a,
        out_b,
        out_c,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        cartesian_case.default_sizes[KDim],
        inout=((out_a, (out_b, out_c)), out_c),
        ref=((a, (b, c)), c),
    )


@gtx.field_operator
def testee_two_vertical_dims(
    a: KField, b: gtx.Field[[KHalfDim], gtx.float32]
) -> tuple[gtx.Field[[KHalfDim], gtx.float32], KField]:
    return b, a


@gtx.program
def prog_two_vertical_dims(
    a: KField,
    b: gtx.Field[[KHalfDim], gtx.float32],
    out_a: KField,
    out_b: gtx.Field[[KHalfDim], gtx.float32],
    k_size: gtx.int32,
    k_half_size: gtx.int32,
):
    testee_two_vertical_dims(
        a, b, out=(out_b, out_a), domain=({KHalfDim: (0, k_half_size)}, {KDim: (0, k_size)})
    )


def test_program_two_vertical_dims(cartesian_case):
    a = cases.allocate(cartesian_case, prog_two_vertical_dims, "a")()
    b = cases.allocate(cartesian_case, prog_two_vertical_dims, "b")()
    out_a = cases.allocate(cartesian_case, prog_two_vertical_dims, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_two_vertical_dims, "out_b")()

    cases.verify(
        cartesian_case,
        prog_two_vertical_dims,
        a,
        b,
        out_a,
        out_b,
        cartesian_case.default_sizes[KDim],
        cartesian_case.default_sizes[KHalfDim],
        inout=(out_b, out_a),
        ref=(b, a),
    )


@gtx.field_operator
def testee_shift_e2c(a: EField) -> tuple[CField, EField]:
    return a(C2E[1]), a


@gtx.program
def prog_unstructured(
    a: EField,
    out_a: EField,
    out_a_shifted: CField,
    c_size: gtx.int32,
    e_size: gtx.int32,
):
    testee_shift_e2c(
        a, out=(out_a_shifted, out_a), domain=({Cell: (0, c_size)}, {Edge: (0, e_size)})
    )


def test_program_unstructured(
    exec_alloc_descriptor, mesh_descriptor
):  # TODO: this fails for definitions_numpy, please see test_temporaries_with_sizes.py
    unstructured_case = Case(
        exec_alloc_descriptor,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )
    a = cases.allocate(unstructured_case, prog_unstructured, "a")()
    out_a = cases.allocate(unstructured_case, prog_unstructured, "out_a")()
    out_a_shifted = cases.allocate(unstructured_case, prog_unstructured, "out_a_shifted")()

    cases.verify(
        unstructured_case,
        prog_unstructured,
        a,
        out_a,
        out_a_shifted,
        unstructured_case.default_sizes[Cell],
        unstructured_case.default_sizes[Edge],
        inout=(out_a_shifted, out_a),
        ref=((a.ndarray)[mesh_descriptor.offset_provider["C2E"].asnumpy()[:, 1]], a),
    )


@gtx.field_operator
def testee_temporary(a: VField):
    edge = a(E2V[1])
    cell = edge(C2E[1])
    return edge, cell


@gtx.program
def prog_temporary(
    a: VField,
    out_edge: EField,
    out_cell: CField,
    c_size: gtx.int32,
    e_size: gtx.int32,
):
    testee_temporary(
        a, out=(out_edge, out_cell), domain=({Edge: (0, e_size)}, {Cell: (0, c_size)})
    )  # TODO: specify other domain sizes?


def test_program_temporary(
    exec_alloc_descriptor, mesh_descriptor
):  # TODO: this fails for definitions_numpy, please see test_temporaries_with_sizes.py
    unstructured_case = Case(
        exec_alloc_descriptor,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
            Vertex: mesh_descriptor.num_vertices,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )
    a = cases.allocate(unstructured_case, prog_temporary, "a")()
    out_edge = cases.allocate(unstructured_case, prog_temporary, "out_edge")()
    out_cell = cases.allocate(unstructured_case, prog_temporary, "out_cell")()

    e2v = (a.ndarray)[mesh_descriptor.offset_provider["E2V"].asnumpy()[:, 1]]
    cases.verify(
        unstructured_case,
        prog_temporary,
        a,
        out_edge,
        out_cell,
        unstructured_case.default_sizes[Cell],
        unstructured_case.default_sizes[Edge],
        inout=(out_edge, out_cell),
        ref=(e2v, e2v[mesh_descriptor.offset_provider["C2E"].asnumpy()[:, 1]]),
    )


def test_direct_fo_orig(cartesian_case):
    a = cases.allocate(cartesian_case, testee_orig, "a")()
    b = cases.allocate(cartesian_case, testee_orig, "b")()
    out = cases.allocate(cartesian_case, testee_orig, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee_orig,
        a,
        b,
        out=out,
        ref=(b, a),
        domain={IDim: (0, cartesian_case.default_sizes[IDim])},
    )


# TODO:
#  - vertical staggering with dependency
#  - cleanup and refactor tests

#
# def test_direct_fo(cartesian_case):
#     a = cases.allocate(cartesian_case, testee, "a")()
#     b = cases.allocate(cartesian_case, testee, "b")()
#     out = cases.allocate(cartesian_case, testee, cases.RETURN)()
#
#     cases.verify(
#         cartesian_case,
#         testee,
#         a,
#         b,
#         out=out,
#         ref=(b, a),
#         domain=(
#             {JDim: (0, cartesian_case.default_sizes[JDim])},
#             {IDim: (0, cartesian_case.default_sizes[IDim])},
#         ),
#     )
