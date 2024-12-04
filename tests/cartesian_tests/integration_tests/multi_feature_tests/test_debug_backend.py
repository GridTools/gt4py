# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from gt4py import storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import BACKWARD, PARALLEL, THIS_K, computation, interval, sin


def test_simple_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(BACKWARD):
            with interval(-2, -1):  # block 1
                field_out = field_in
            with interval(0, -2):  # block 2
                field_out = field_in
        with computation(BACKWARD):
            with interval(-1, None):  # block 3
                field_out = 2 * field_in
            with interval(0, -1):  # block 4
                field_out = 3 * field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0:-1], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, -1], 2)


def test_tmp_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(PARALLEL):
            with interval(...):
                tmp = field_in + 1
        with computation(PARALLEL):
            with interval(...):
                field_out = tmp[-1, 0, 0] + tmp[1, 0, 0]

    stencil(field_in, field_out, origin=(1, 1, 0), domain=(4, 4, 6))

    # the inside of the domain is 4
    np.testing.assert_allclose(field_out.view(np.ndarray)[1:-1, 1:-1, :], 4)
    # the rest is 0
    np.testing.assert_allclose(field_out.view(np.ndarray)[0:1, :, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[-1:, :, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, 0:1, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, -1:, :], 0)


def test_backward_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(BACKWARD):
            with interval(-1, None):
                field_in = 2
                field_out = field_in
            with interval(0, -1):
                field_in = field_in[0, 0, 1] + 1
                field_out = field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0], 5)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 1], 4)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 2], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 3], 2)


def test_while_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(PARALLEL):
            with interval(...):
                while field_in < 10:
                    field_in += 1
                field_out = field_in

    stencil(field_in, field_out)

    # the inside of the domain is 10
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], 10)


def test_higher_dim_literal_stencil():
    FLOAT64_NDDIM = (np.float64, (4,))

    field_in = gt_storage.ones(
        dtype=FLOAT64_NDDIM, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_in[:, :, :, 2] = 5

    @gtscript.stencil(backend="debug")
    def stencil(
        vec_field: gtscript.Field[FLOAT64_NDDIM],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = vec_field[0, 0, 0][2]

    stencil(field_in, field_out)

    # the inside of the domain is 5
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], 5)


def test_higher_dim_scalar_stencil():
    FLOAT64_NDDIM = (np.float64, (4,))

    field_in = gt_storage.ones(
        dtype=FLOAT64_NDDIM, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_in[:, :, :, 2] = 5

    @gtscript.stencil(backend="debug")
    def stencil(
        vec_field: gtscript.Field[FLOAT64_NDDIM],
        out_field: gtscript.Field[np.float64],
        scalar_argument: int,
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = vec_field[0, 0, 0][scalar_argument]

    stencil(field_in, field_out, 2)

    # the inside of the domain is 5
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], 5)


def test_native_function_call_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 0] + sin(0.848062)

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], 1.75)


def test_unary_operator_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = -in_field[0, 0, 0]

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], -1)


def test_ternary_operator_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[0, 0, 1] = 20

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 0] if in_field > 10 else in_field[0, 0, 0] + 1

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[0, 0, 1], 20)
    np.testing.assert_allclose(field_out.view(np.ndarray)[1:, 1:, 1], 2)


def test_mask_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[0, 0, 1] = -20

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            if in_field[0, 0, 0] > 0:
                out_field[0, 0, 0] = in_field
            else:
                out_field[0, 0, 0] = 1

    test_stencil(field_in, field_out)

    assert np.all(field_out.view(np.ndarray) > 0)


def test_k_offset_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:, :, 0] *= 10
    offset = -1

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
        scalar_value: int,
    ):
        with computation(PARALLEL), interval(1, None):
            out_field[0, 0, 0] = in_field[0, 0, scalar_value]

    test_stencil(field_in, field_out, offset)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 1], 10)


def test_k_offset_field_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_idx = gt_storage.ones(dtype=np.int64, backend="debug", shape=(4, 4), aligned_index=(0, 0))
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:, :, 0] *= 10
    field_idx[:, :] *= -2

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
        idx_field: gtscript.Field[gtscript.IJ, np.int64],
    ):
        with computation(PARALLEL), interval(1, None):
            out_field[0, 0, 0] = in_field[0, 0, idx_field + 1]

    test_stencil(field_in, field_out, field_idx)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 1], 10)


def test_absolute_k_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:, :, 0] *= 10
    field_in[:, :, 1] *= 5

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field.at(K=0) + in_field.at(K=1)

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, :], 15)


def test_k_only_access_stencil():
    field_in = np.ones((4,), dtype=np.float64)
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:] = [2, 3, 4, 5]

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[gtscript.K, np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL):
            with interval(0, 1):
                out_field[0, 0, 0] = in_field[1]
            with interval(1, None):
                out_field[0, 0, 0] = in_field[-1]

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[1, 1, :], [3, 2, 3, 4])


def test_table_access_stencil():
    table_view = np.ones((4,), dtype=np.float64)
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    table_view[:] = [2, 3, 4, 5]

    @gtscript.stencil(backend="debug")
    def test_stencil(
        table_view: gtscript.GlobalTable[(np.float64, (4))],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL):
            with interval(0, 1):
                out_field[0, 0, 0] = table_view.A[1]
            with interval(1, None):
                out_field[0, 0, 0] = table_view.A[2]

    test_stencil(table_view, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[1, 1, :], [3, 4, 4, 4])

def test_this_k_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def test_stencil(
        in_field: gtscript.Field[np.float64],
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            tmp = THIS_K
            out_field[0, 0, 0] = in_field.at(K=tmp)

    test_stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 1], 1)
