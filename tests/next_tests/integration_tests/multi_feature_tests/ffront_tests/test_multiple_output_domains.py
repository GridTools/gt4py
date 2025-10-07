# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np
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
    cartesian_case,
    unstructured_case,
    Case,
    IField,
    JField,
    KField,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)

KHalfDim = gtx.Dimension("KHalf", kind=gtx.DimensionKind.VERTICAL)
pytestmark = pytest.mark.uses_cartesian_shift


@gtx.field_operator
def testee_no_tuple(a: IField, b: JField) -> IField:
    return a


@gtx.program
def prog_no_tuple(
    a: IField,
    b: JField,
    out_a: IField,
    i_size: gtx.int32,
):
    testee_no_tuple(a, b, out=out_a, domain={IDim: (0, i_size)})


def test_program_no_tuple(cartesian_case):
    a = cases.allocate(cartesian_case, prog_no_tuple, "a")()
    b = cases.allocate(cartesian_case, prog_no_tuple, "b")()
    out_a = cases.allocate(cartesian_case, prog_no_tuple, "out_a")()

    cases.verify(
        cartesian_case,
        prog_no_tuple,
        a,
        b,
        out_a,
        cartesian_case.default_sizes[IDim],
        inout=out_a,
        ref=a,
    )


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
def prog_no_domain_different_fields(
    a: IField,
    b: JField,
    out_a: IField,
    out_b: JField,
):
    testee(a, b, out=(out_b, out_a))


def test_program_no_domain_different_fields(
    cartesian_case,
):
    a = cases.allocate(cartesian_case, prog_no_domain_different_fields, "a")()
    b = cases.allocate(cartesian_case, prog_no_domain_different_fields, "b")()
    out_a = cases.allocate(cartesian_case, prog_no_domain_different_fields, "out_a")()
    out_b = cases.allocate(cartesian_case, prog_no_domain_different_fields, "out_b")()

    cases.verify(
        cartesian_case,
        prog_no_domain_different_fields,
        a,
        b,
        out_a,
        out_b,
        inout=(out_b, out_a),
        ref=(b, a),
    )


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
def prog_slicing(
    a: IField,
    b: JField,
    out_a: IField,
    out_b: JField,
):
    testee(
        a,
        b,
        out=(out_b[2:-2], out_a[1:-1]),
    )


def test_program_slicing(cartesian_case):
    a = cases.allocate(cartesian_case, prog, "a")()
    b = cases.allocate(cartesian_case, prog, "b")()
    out_a = cases.allocate(cartesian_case, prog, "out_a")()
    out_b = cases.allocate(cartesian_case, prog, "out_b")()
    out_a_ = copy.deepcopy(out_a)
    out_b_ = copy.deepcopy(out_b)
    cases.verify(
        cartesian_case,
        prog_slicing,
        a,
        b,
        out_a,
        out_b,
        inout=(out_b, out_a),
        ref=(
            np.concatenate([out_b_.ndarray[0:2], b.ndarray[2:-2], out_b_.ndarray[-2:]]),
            np.concatenate([out_a_.ndarray[0:1], a.ndarray[1:-1], out_a_.ndarray[-1:]]),
        ),
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
):
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


@gtx.program
def prog_out_as_tuple_different_sizes(
    a: IField,
    b: JField,
    out: tuple[JField, IField],
    i_size: gtx.int32,
    j_size: gtx.int32,
    restrict_i_0: gtx.int32,
    restrict_i_1: gtx.int32,
    restrict_j_0: gtx.int32,
    restrict_j_1: gtx.int32,
):
    testee(
        a,
        b,
        out=out,
        domain=(
            {JDim: (restrict_j_0, j_size + restrict_j_1)},
            {IDim: (restrict_i_0, i_size + restrict_i_1)},
        ),
    )


def test_program_out_as_tuple_different_sizes(
    cartesian_case,
):
    restrict_i = (1, -3)
    restrict_j = (2, -4)
    i_size = cartesian_case.default_sizes[IDim]
    j_size = cartesian_case.default_sizes[JDim]
    a = cases.allocate(cartesian_case, prog_out_as_tuple_different_sizes, "a")()
    b = cases.allocate(cartesian_case, prog_out_as_tuple_different_sizes, "b")()
    out = cases.allocate(
        cartesian_case,
        prog_out_as_tuple_different_sizes,
        "out",
        extend={IDim: (-restrict_i[0], restrict_i[1]), JDim: (-restrict_j[0], restrict_j[1])},
    )()

    cases.verify(
        cartesian_case,
        prog_out_as_tuple_different_sizes,
        a,
        b,
        out,
        i_size,
        j_size,
        restrict_i[0],
        restrict_i[1],
        restrict_j[0],
        restrict_j[1],
        inout=(out),
        ref=(
            b.ndarray[restrict_j[0] : j_size + restrict_j[1]],
            a.ndarray[restrict_i[0] : i_size + restrict_i[1]],
        ),
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
    out_c0: KField,
    out_c1: KField,
    i_size: gtx.int32,
    j_size: gtx.int32,
    k_size: gtx.int32,
):
    testee_double_nested_tuples(
        a,
        b,
        c,
        out=((out_a, (out_b, out_c0)), out_c1),
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
    out_c0 = cases.allocate(cartesian_case, prog_double_nested_tuples, "out_c0")()
    out_c1 = cases.allocate(cartesian_case, prog_double_nested_tuples, "out_c1")()

    cases.verify(
        cartesian_case,
        prog_double_nested_tuples,
        a,
        b,
        c,
        out_a,
        out_b,
        out_c0,
        out_c1,
        cartesian_case.default_sizes[IDim],
        cartesian_case.default_sizes[JDim],
        cartesian_case.default_sizes[KDim],
        inout=((out_a, (out_b, out_c0)), out_c1),
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


def test_program_unstructured(unstructured_case):
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
        ref=((a.ndarray)[unstructured_case.offset_provider["C2E"].asnumpy()[:, 1]], a),
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
    restrict_edge_0: gtx.int32,
    restrict_edge_1: gtx.int32,
    restrict_cell_0: gtx.int32,
    restrict_cell_1: gtx.int32,
):
    testee_temporary(
        a,
        out=(out_edge, out_cell),
        domain=(
            {Edge: (restrict_edge_0, e_size + restrict_edge_1)},
            {Cell: (restrict_cell_0, c_size + restrict_cell_1)},
        ),
    )


def test_program_temporary(unstructured_case):
    restrict_edge = (4, -2)
    restrict_cell = (3, -1)
    cell_size = unstructured_case.default_sizes[Cell]
    edge_size = unstructured_case.default_sizes[Edge]
    a = cases.allocate(unstructured_case, prog_temporary, "a")()
    out_edge = cases.allocate(
        unstructured_case,
        prog_temporary,
        "out_edge",
        extend={Edge: (-restrict_edge[0], restrict_edge[1])},
    )()
    out_cell = cases.allocate(
        unstructured_case,
        prog_temporary,
        "out_cell",
        extend={Cell: (-restrict_cell[0], restrict_cell[1])},
    )()

    e2v = (a.ndarray)[unstructured_case.offset_provider["E2V"].asnumpy()[:, 1]]
    cases.verify(
        unstructured_case,
        prog_temporary,
        a,
        out_edge,
        out_cell,
        cell_size,
        edge_size,
        restrict_edge[0],
        restrict_edge[1],
        restrict_cell[0],
        restrict_cell[1],
        inout=(out_edge, out_cell),
        ref=(
            e2v[restrict_edge[0] : edge_size + restrict_edge[1]],
            e2v[unstructured_case.offset_provider["C2E"].asnumpy()[:, 1]][
                restrict_cell[0] : cell_size + restrict_cell[1]
            ],
        ),
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


def test_direct_fo_nested(cartesian_case):
    a = cases.allocate(cartesian_case, testee_nested_tuples, "a")()
    b = cases.allocate(cartesian_case, testee_nested_tuples, "b")()
    c = cases.allocate(cartesian_case, testee_nested_tuples, "c")()
    out = cases.allocate(cartesian_case, testee_nested_tuples, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee_nested_tuples,
        a,
        b,
        c,
        out=out,
        ref=((a, b), c),
        domain=(
            (
                {IDim: (0, cartesian_case.default_sizes[IDim])},
                {JDim: (0, cartesian_case.default_sizes[JDim])},
            ),
            {KDim: (0, cartesian_case.default_sizes[KDim])},
        ),
    )


def test_direct_fo(cartesian_case):
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        out=out,
        ref=(b, a),
        domain=(
            {JDim: (0, cartesian_case.default_sizes[JDim])},
            {IDim: (0, cartesian_case.default_sizes[IDim])},
        ),
    )


def test_direct_fo_nested_no_domain(cartesian_case):
    a = cases.allocate(cartesian_case, testee_nested_tuples, "a")()
    b = cases.allocate(cartesian_case, testee_nested_tuples, "b")()
    c = cases.allocate(cartesian_case, testee_nested_tuples, "c")()
    out = cases.allocate(cartesian_case, testee_nested_tuples, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee_nested_tuples,
        a,
        b,
        c,
        out=out,
        ref=((a, b), c),
    )
