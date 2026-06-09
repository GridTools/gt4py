# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, float64, int32, max_over, min_over, neighbor_sum, where
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    Edge,
    JDim,
    Joff,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_max_over
@pytest.mark.parametrize(
    "strategy",
    [cases.UniqueInitializer(1), cases.UniqueInitializer(-100)],
    ids=["positive_values", "negative_values"],
)
def test_maxover_execution_(unstructured_case, strategy):
    @gtx.field_operator
    def testee(edge_f: cases.EField) -> cases.VField:
        out = max_over(edge_f(V2E), axis=V2EDim)
        return out

    inp = cases.allocate(unstructured_case, testee, "edge_f", strategy=strategy)()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    ref = np.max(
        inp.asnumpy()[v2e_table],
        axis=1,
        initial=np.min(inp.asnumpy()),
        where=v2e_table != common._DEFAULT_SKIP_VALUE,
    )
    cases.verify(unstructured_case, testee, inp, ref=ref, out=out)


@pytest.mark.uses_unstructured_shift
def test_minover_execution(unstructured_case):
    @gtx.field_operator
    def minover(edge_f: cases.EField) -> cases.VField:
        out = min_over(edge_f(V2E), axis=V2EDim)
        return out

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        minover,
        ref=lambda edge_f: np.min(
            edge_f[v2e_table],
            axis=1,
            initial=np.max(edge_f),
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@gtx.field_operator
def reduction_e_field(edge_f: cases.EField) -> cases.VField:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@gtx.field_operator
def reduction_ek_field(
    edge_f: common.Field[[Edge, KDim], np.int32],
) -> common.Field[[Vertex, KDim], np.int32]:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@pytest.mark.uses_unstructured_shift
@pytest.mark.parametrize(
    "fop", [reduction_e_field, reduction_ek_field], ids=lambda fop: fop.__name__
)
def test_neighbor_sum(unstructured_case_3d, fop):
    v2e_table = unstructured_case_3d.offset_provider["V2E"].asnumpy()

    edge_f = cases.allocate(unstructured_case_3d, fop, "edge_f")()

    local_dim_idx = edge_f.domain.dims.index(Edge) + 1
    adv_indexing = tuple(
        slice(None) if dim is not Edge else v2e_table for dim in edge_f.domain.dims
    )

    broadcast_slice = []
    for dim in edge_f.domain.dims:
        if dim is Edge:
            broadcast_slice.append(slice(None))
            broadcast_slice.append(slice(None))
        else:
            broadcast_slice.append(None)

    broadcasted_table = v2e_table[tuple(broadcast_slice)]
    ref = np.sum(
        edge_f.asnumpy()[adv_indexing],
        axis=local_dim_idx,
        initial=0,
        where=broadcasted_table != common._DEFAULT_SKIP_VALUE,
    )
    cases.verify(
        unstructured_case_3d,
        fop,
        edge_f,
        out=cases.allocate(unstructured_case_3d, fop, cases.RETURN)(),
        ref=ref,
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_execution_with_offset(unstructured_case_3d):
    EKField: TypeAlias = gtx.Field[[Edge, KDim], np.int32]
    VKField: TypeAlias = gtx.Field[[Vertex, KDim], np.int32]

    @gtx.field_operator
    def reduction(edge_f: EKField) -> VKField:
        return neighbor_sum(edge_f(V2E), axis=V2EDim)

    @gtx.field_operator
    def fencil_op(edge_f: EKField) -> VKField:
        red = reduction(edge_f)
        return red(Koff[1])

    @gtx.program
    def fencil(edge_f: EKField, out: VKField):
        fencil_op(edge_f, out=out)

    v2e_table = unstructured_case_3d.offset_provider["V2E"].asnumpy()
    field = cases.allocate(unstructured_case_3d, fencil, "edge_f", sizes={KDim: 2})()
    out = cases.allocate(unstructured_case_3d, fencil_op, cases.RETURN, sizes={KDim: 1})()

    cases.verify(
        unstructured_case_3d,
        fencil,
        field,
        out,
        inout=out,
        ref=np.sum(
            field.asnumpy()[:, 1][v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ).reshape(out.shape),
        offset_provider=unstructured_case_3d.offset_provider,
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_constant_fields
def test_reduction_expression_in_call(unstructured_case):
    @gtx.field_operator
    def reduce_expr(edge_f: cases.EField) -> cases.VField:
        tmp_nbh_tup = edge_f(V2E), edge_f(V2E)
        tmp_nbh = tmp_nbh_tup[0]
        return 3 * neighbor_sum(-edge_f(V2E) * tmp_nbh * 2, axis=V2EDim)

    @gtx.program
    def fencil(edge_f: cases.EField, out: cases.VField):
        reduce_expr(edge_f, out=out)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        fencil,
        ref=lambda edge_f: (
            3
            * np.sum(
                -(edge_f[v2e_table] ** 2) * 2,
                axis=1,
                initial=0,
                where=v2e_table != common._DEFAULT_SKIP_VALUE,
            )
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_with_common_expression(unstructured_case):
    @gtx.field_operator
    def testee(flux: cases.EField) -> cases.VField:
        return neighbor_sum(flux(V2E) + flux(V2E), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda flux: np.sum(
            flux[v2e_table] * 2, axis=1, initial=0, where=v2e_table != common._DEFAULT_SKIP_VALUE
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(where(mask, inp(V2E), inp(V2E)), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where_and_tuples(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(where(mask, (inp(V2E), inp(V2E)), (inp(V2E), inp(V2E)))[1], axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table],
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


@pytest.mark.uses_unstructured_shift
def test_reduction_expression_with_where_and_scalar(unstructured_case):
    @gtx.field_operator
    def testee(mask: cases.VBoolField, inp: cases.EField) -> cases.VField:
        return neighbor_sum(inp(V2E) + where(mask, inp(V2E), 1), axis=V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()

    mask = unstructured_case.as_field(
        [Vertex], np.random.choice(a=[False, True], size=unstructured_case.default_sizes[Vertex])
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    cases.verify(
        unstructured_case,
        testee,
        mask,
        inp,
        out=out,
        ref=np.sum(
            inp.asnumpy()[v2e_table]
            + np.where(np.expand_dims(mask.asnumpy(), 1), inp.asnumpy()[v2e_table], 1),
            axis=1,
            initial=0,
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )


def test_promotion(unstructured_case_3d):
    @gtx.field_operator
    def promotion(
        inp1: gtx.Field[[Edge, KDim], float64], inp2: gtx.Field[[KDim], float64]
    ) -> gtx.Field[[Edge, KDim], float64]:
        return inp1 / inp2

    cases.verify_with_default_data(
        unstructured_case_3d, promotion, ref=lambda inp1, inp2: inp1 / inp2
    )


@pytest.mark.uses_unstructured_shift
def test_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]],
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_program_with_sliced_out_arguments
def test_unstructured_shift_with_non_zero_origin(unstructured_case):
    # TODO(edopao,havogt): remove the xfail below once the embedded backend supports
    #   non-contiguous field domain, which is generated by the inverse image of the
    #   connectivity used in this test.
    if unstructured_case.backend is None:
        pytest.xfail("Embedded backend only supports contiguous field domains.")

    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    a = cases.allocate(unstructured_case, testee, "a")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    ORIGIN = 2
    e2v_table = unstructured_case.offset_provider["E2V"].asnumpy()
    neighbor_0_iter = iter(enumerate(e2v_table[:, 0]))
    edge_start = next(i for i, v in neighbor_0_iter if v >= ORIGIN)
    edge_stop = next(i for i, v in neighbor_0_iter if v < ORIGIN)

    ref = a.ndarray[e2v_table[edge_start:edge_stop, 0]]
    cases.verify(unstructured_case, testee, a[ORIGIN:], out=out[edge_start:edge_stop], ref=ref)


def test_horizontal_only_with_3d_mesh(unstructured_case_3d):
    # test field operator operating only on horizontal fields while using an offset provider
    # including a vertical dimension.
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.VField:
        return a

    cases.verify_with_default_data(
        unstructured_case_3d,
        testee,
        ref=lambda a: a,
    )


@pytest.mark.uses_unstructured_shift
def test_composed_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def composed_shift_unstructured_flat(inp: cases.VField) -> cases.CField:
        return inp(E2V[0])(C2E[0])

    @gtx.field_operator
    def composed_shift_unstructured_intermediate_result(inp: cases.VField) -> cases.CField:
        tmp = inp(E2V[0])
        return tmp(C2E[0])

    @gtx.field_operator
    def shift_e2v(inp: cases.VField) -> cases.EField:
        return inp(E2V[0])

    @gtx.field_operator
    def composed_shift_unstructured(inp: cases.VField) -> cases.CField:
        return shift_e2v(inp)(C2E[0])

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_flat,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_intermediate_result,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
        comparison=lambda inp, tmp: np.all(inp == tmp),
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]][
            unstructured_case.offset_provider["C2E"].asnumpy()[:, 0]
        ],
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_program_with_sliced_out_arguments
def test_neighbor_sum_with_non_zero_origin(unstructured_case):
    # TODO(edopao,havogt): remove the xfail below once the embedded backend supports
    #   non-contiguous field domain, which is generated by the inverse image of the
    #   connectivity used in this test.
    if unstructured_case.backend is None:
        pytest.xfail("Embedded backend only supports contiguous field domains.")

    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return neighbor_sum(a(E2V), axis=E2VDim)

    a = cases.allocate(unstructured_case, testee, "a")()
    out = cases.allocate(unstructured_case, testee, cases.RETURN)()

    ORIGIN = 2
    e2v_table = unstructured_case.offset_provider["E2V"].asnumpy()
    neighbor_iter = iter(enumerate(e2v_table))
    edge_start = next(i for i, v in neighbor_iter if all(v >= ORIGIN))
    edge_stop = next(i for i, v in neighbor_iter if any(v < ORIGIN))

    ref = np.sum(a.ndarray[e2v_table[edge_start:edge_stop,]], axis=1)
    cases.verify(unstructured_case, testee, a[ORIGIN:], out=out[edge_start:edge_stop], ref=ref)


@pytest.mark.uses_unstructured_shift
def test_nested_reduction(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.VField:
        tmp = neighbor_sum(a(E2V), axis=E2VDim)
        tmp_2 = neighbor_sum(tmp(V2E), axis=V2EDim)
        return tmp_2

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(
            np.sum(a[unstructured_case.offset_provider["E2V"].asnumpy()], axis=1, initial=0)[
                unstructured_case.offset_provider["V2E"].asnumpy()
            ],
            axis=1,
            where=unstructured_case.offset_provider["V2E"].asnumpy() != common._DEFAULT_SKIP_VALUE,
        ),
        comparison=lambda a, tmp_2: np.all(a == tmp_2),
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.xfail(reason="Not yet supported in lowering, requires `map_`ing of inner reduce op.")
def test_nested_reduction_shift_first(unstructured_case):
    @gtx.field_operator
    def testee(inp: cases.EField) -> cases.EField:
        tmp = inp(V2E)
        tmp2 = tmp(E2V)
        return neighbor_sum(neighbor_sum(tmp2, axis=V2EDim), axis=E2VDim)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda inp: np.sum(
            np.sum(inp[unstructured_case.offset_provider["V2E"].asnumpy()], axis=1)[
                unstructured_case.offset_provider["E2V"].asnumpy()
            ],
            axis=1,
        ),
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_constant_fields
def test_tuple_with_local_field_in_reduction_shifted(unstructured_case):
    @gtx.field_operator
    def reduce_tuple_element(e: cases.EField, v: cases.VField) -> cases.EField:
        tup = e(V2E), v
        red = neighbor_sum(tup[0] + v, axis=V2EDim)
        tmp = red(E2V[0])
        return tmp

    v2e = unstructured_case.offset_provider["V2E"]
    cases.verify_with_default_data(
        unstructured_case,
        reduce_tuple_element,
        ref=lambda e, v: np.sum(
            e[v2e.asnumpy()] + np.tile(v, (v2e.shape[1], 1)).T,
            axis=1,
            initial=0,
            where=v2e.asnumpy() != common._DEFAULT_SKIP_VALUE,
        )[unstructured_case.offset_provider["E2V"].asnumpy()[:, 0]],
    )


@pytest.mark.uses_constant_fields
@pytest.mark.uses_unstructured_shift
def test_ternary_builtin_neighbor_sum(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField, b: cases.EField) -> cases.VField:
        tmp = neighbor_sum(b(V2E) if 2 < 3 else a(V2E), axis=V2EDim)
        return tmp

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a, b: np.sum(
            b[v2e_table], axis=1, initial=0, where=v2e_table != common._DEFAULT_SKIP_VALUE
        ),
    )


def test_local_index_premapped_field(request, unstructured_case):
    if request.node.get_closest_marker(pytest.mark.uses_mesh_with_skip_values.name):
        pytest.skip("This test only works with non-skip value meshes.")

    @gtx.field_operator
    def testee(inp: gtx.Field[[Edge], int32]) -> gtx.Field[[Vertex], int32]:
        shifted = inp(V2E)
        return shifted[V2EDim(0)] + shifted[V2EDim(1)] + shifted[V2EDim(2)] + shifted[V2EDim(3)]

    inp = cases.allocate(unstructured_case, testee, "inp")()

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify(
        unstructured_case,
        testee,
        inp,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(inp.asnumpy()[v2e_table], axis=1),
    )
