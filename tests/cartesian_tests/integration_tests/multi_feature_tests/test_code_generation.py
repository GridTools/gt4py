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

import numpy as np
import pytest

from gt4py import storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    Field,
    GlobalTable,
    I,
    J,
    computation,
    horizontal,
    interval,
    region,
)
from gt4py.storage.cartesian import utils as storage_utils

from cartesian_tests.definitions import ALL_BACKENDS, CPU_BACKENDS
from cartesian_tests.integration_tests.multi_feature_tests.stencil_definitions import (
    EXTERNALS_REGISTRY as externals_registry,
    REGISTRY as stencil_definitions,
)


@pytest.mark.parametrize("name", stencil_definitions)
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_generation(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=(v.dtype, v.data_dims) if v.data_dims else v.dtype,
                dimensions=v.axes,
                backend=backend,
                shape=(23, 23, 23),
                aligned_index=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    # vertical domain size >= 16 required for test_large_k_interval
    stencil(**args, origin=(10, 10, 5), domain=(3, 3, 16))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lazy_stencil(backend):
    @gtscript.lazy_stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_], field_b: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            field_a = field_b


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_temporary_field_declared_in_if(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            if field_a < 0:
                field_b = -field_a
            else:
                field_b = field_a
            field_a = field_b


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_stage_without_effect(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            field_c = 0.0


def test_ignore_np_errstate():
    def setup_and_run(backend, **kwargs):
        field_a = gt_storage.zeros(
            dtype=np.float_, backend=backend, shape=(3, 3, 1), aligned_index=(0, 0, 0)
        )

        @gtscript.stencil(backend=backend, **kwargs)
        def divide_by_zero(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                field_a = 1.0 / field_a

        divide_by_zero(field_a)

    # Usual behavior: with the numpy backend there is no error
    setup_and_run(backend="numpy")

    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="numpy", ignore_np_errstate=False)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stencil_without_effect(backend):
    def definition1(field_in: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            tmp = 0.0

    def definition2(f_in: gtscript.Field[np.float_]):
        from __externals__ import flag

        with computation(PARALLEL), interval(...):
            if __INLINED(flag):
                B = f_in

    stencil1 = gtscript.stencil(backend, definition1)
    stencil2 = gtscript.stencil(backend, definition2, externals={"flag": False})

    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )

    # test with explicit domain specified
    stencil1(field_in, domain=(3, 3, 3))
    stencil2(field_in, domain=(3, 3, 3))

    # test without domain specified
    stencil1(field_in)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_merger_induced_interval_block_reordering(backend):
    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float_], field_out: gtscript.Field[np.float_]):
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


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_inputs(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        field_3d: gtscript.Field[gtscript.IJK, np.float_],
        field_2d: gtscript.Field[gtscript.IJ, np.float_],
        field_1d: gtscript.Field[gtscript.K, np.float_],
    ):
        with computation(PARALLEL):
            with interval(0, -1):
                tmp = field_2d + field_1d[1]
            with interval(-1, None):
                tmp = field_2d + field_1d[0]

        with computation(PARALLEL):
            with interval(0, 1):
                field_3d = tmp[1, 0, 0] + field_1d[1]
            with interval(1, None):
                field_3d = tmp[-1, 0, 0]

    full_shape = (6, 6, 6)
    aligned_index = (1, 1, 0)
    dtype = float

    field_3d = gt_storage.zeros(
        full_shape, dtype, backend=backend, aligned_index=aligned_index, dimensions=None
    )
    assert field_3d.shape == full_shape[:]

    field_2d = gt_storage.zeros(
        full_shape[:-1], dtype, backend=backend, aligned_index=aligned_index[:-1], dimensions="IJ"
    )
    assert field_2d.shape == full_shape[:-1]

    field_1d = gt_storage.ones(
        full_shape[-1:], dtype, backend=backend, aligned_index=(aligned_index[-1],), dimensions="K"
    )
    assert list(field_1d.shape) == [full_shape[-1]]

    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0), domain=(4, 3, 6))
    res_field_3d = storage_utils.cpu_copy(field_3d)
    np.testing.assert_allclose(res_field_3d[1:-1, 1:-2, :1], 2)
    np.testing.assert_allclose(res_field_3d[1:-1, 1:-2, 1:], 1)

    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_masked(backend):
    @gtscript.stencil(backend=backend)
    def copy_2to3(
        cond: gtscript.Field[gtscript.IJK, np.float_],
        inp: gtscript.Field[gtscript.IJ, np.float_],
        outp: gtscript.Field[gtscript.IJK, np.float_],
    ):
        with computation(PARALLEL), interval(...):
            if cond > 0.0:
                outp = inp

    inp = np.random.randn(10, 10)
    outp = np.random.randn(10, 10, 10)
    cond = np.random.randn(10, 10, 10)

    inp_f = gt_storage.from_array(inp, aligned_index=(0, 0), backend=backend)
    outp_f = gt_storage.from_array(outp, aligned_index=(0, 0, 0), backend=backend)
    cond_f = gt_storage.from_array(cond, aligned_index=(0, 0, 0), backend=backend)

    copy_2to3(cond_f, inp_f, outp_f)

    inp3d = np.empty_like(outp)
    inp3d[...] = inp[:, :, np.newaxis]

    outp = np.choose(cond > 0.0, [outp, inp3d])

    outp_f = storage_utils.cpu_copy(outp)
    assert np.allclose(outp, outp_f)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_masked_2dcond(backend):
    @gtscript.stencil(backend=backend)
    def copy_2to3(
        cond: gtscript.Field[gtscript.IJK, np.float_],
        inp: gtscript.Field[gtscript.IJ, np.float_],
        outp: gtscript.Field[gtscript.IJK, np.float_],
    ):
        with computation(FORWARD), interval(...):
            if cond > 0.0:
                outp = inp

    inp = np.random.randn(10, 10)
    outp = np.random.randn(10, 10, 10)
    cond = np.random.randn(10, 10, 10)

    inp_f = gt_storage.from_array(inp, aligned_index=(0, 0), backend=backend)
    outp_f = gt_storage.from_array(outp, aligned_index=(0, 0, 0), backend=backend)
    cond_f = gt_storage.from_array(cond, aligned_index=(0, 0, 0), backend=backend)

    copy_2to3(cond_f, inp_f, outp_f)

    inp3d = np.empty_like(outp)
    inp3d[...] = inp[:, :, np.newaxis]

    outp = np.choose(cond > 0.0, [outp, inp3d])

    outp_f = storage_utils.cpu_copy(outp_f)
    assert np.allclose(outp, np.asarray(outp_f))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_inputs_2d_to_3d_forward(backend):
    @gtscript.stencil(backend=backend)
    def copy_2to3(
        inp: gtscript.Field[gtscript.IJ, np.float_], outp: gtscript.Field[gtscript.IJK, np.float_]
    ):
        with computation(FORWARD), interval(...):
            outp[0, 0, 0] = inp

    inp_f = gt_storage.from_array(np.random.randn(10, 10), aligned_index=(0, 0), backend=backend)
    outp_f = gt_storage.from_array(
        np.random.randn(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
    )
    copy_2to3(inp_f, outp_f)
    inp_f = storage_utils.cpu_copy(inp_f)
    outp_f = storage_utils.cpu_copy(outp_f)
    assert np.allclose(outp_f, inp_f[:, :, np.newaxis])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_higher_dimensional_fields(backend):
    FLOAT64_VEC2 = (np.float64, (2,))
    FLOAT64_MAT22 = (np.float64, (2, 2))

    @gtscript.stencil(backend=backend)
    def stencil(
        field: gtscript.Field[np.float64],
        vec_field: gtscript.Field[FLOAT64_VEC2],
        mat_field: gtscript.Field[FLOAT64_MAT22],
    ):
        with computation(PARALLEL), interval(...):
            tmp = vec_field[0, 0, 0][0] + vec_field[0, 0, 0][1]

        with computation(FORWARD):
            with interval(0, 1):
                vec_field[0, 0, 0][0] = field[1, 0, 0]
                vec_field[0, 0, 0][1] = field[0, 1, 0]
            with interval(1, -1):
                vec_field[0, 0, 0][0] = 2 * field[1, 0, -1]
                vec_field[0, 0, 0][1] = 2 * field[0, 1, -1]
            with interval(-1, None):
                vec_field[0, 0, 0][0] = field[1, 0, 0]
                vec_field[0, 0, 0][1] = field[0, 1, 0]

        with computation(PARALLEL), interval(...):
            mat_field[0, 0, 0][0, 0] = vec_field[0, 0, 0][0] + 1.0
            mat_field[0, 0, 0][1, 1] = vec_field[0, 0, 0][1] + 1.0

    full_shape = (6, 6, 6)
    aligned_index = (1, 1, 0)

    field = gt_storage.ones(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=np.float64
    )
    assert field.shape == full_shape[:]

    vec_field = gt_storage.ones(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=FLOAT64_VEC2
    )
    vec_field[:] = 2.0
    assert vec_field.shape[:-1] == full_shape

    mat_field = gt_storage.ones(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=FLOAT64_MAT22
    )
    assert mat_field.shape[:-2] == full_shape

    stencil(field, vec_field, mat_field, origin=(1, 1, 0), domain=(4, 4, 6))
    res_mat_field = storage_utils.cpu_copy(mat_field)
    np.testing.assert_allclose(res_mat_field[1:-1, 1:-1, 1:1], 2.0 + 5.0)

    stencil(field, vec_field, mat_field)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_input_order(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        in_field: gtscript.Field[np.float64],
        parameter: np.float64,
        out_field: gtscript.Field[np.float64],
    ):
        with computation(PARALLEL), interval(...):
            out_field = in_field * parameter


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_variable_offsets(backend):
    if backend == "dace:cpu":
        pytest.skip("Internal compiler error in GitHub action container")

    @gtscript.stencil(backend=backend)
    def stencil_ij(
        in_field: gtscript.Field[np.float_],
        out_field: gtscript.Field[np.float_],
        index_field: gtscript.Field[gtscript.IJ, int],
    ):
        with computation(FORWARD), interval(...):
            out_field = in_field[0, 0, 1] + in_field[0, 0, index_field + 1]
            index_field = index_field + 1

    @gtscript.stencil(backend=backend)
    def stencil_ijk(
        in_field: gtscript.Field[np.float_],
        out_field: gtscript.Field[np.float_],
        index_field: gtscript.Field[int],
    ):
        with computation(PARALLEL), interval(...):
            out_field = in_field[0, 0, 1] + in_field[0, 0, index_field + 1]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_variable_offsets_and_while_loop(backend):
    if backend == "dace:cpu":
        pytest.skip("Internal compiler error in GitHub action container")

    @gtscript.stencil(backend=backend)
    def stencil(
        pe1: gtscript.Field[np.float_],
        pe2: gtscript.Field[np.float_],
        qin: gtscript.Field[np.float_],
        qout: gtscript.Field[np.float_],
        lev: gtscript.Field[gtscript.IJ, np.int_],
    ):
        with computation(FORWARD), interval(0, -1):
            if pe2[0, 0, 1] <= pe1[0, 0, lev]:
                qout = qin[0, 0, 1]
            else:
                qsum = pe1[0, 0, lev + 1] - pe2[0, 0, lev]
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += qin[0, 0, lev] / (pe2[0, 0, 1] - pe1[0, 0, lev])
                    lev = lev + 1
                qout = qsum / (pe2[0, 0, 1] - pe2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_nested_while_loop(backend):
    @gtscript.stencil(backend=backend)
    def stencil(field_a: gtscript.Field[np.float_], field_b: gtscript.Field[np.int_]):
        with computation(PARALLEL), interval(...):
            while field_a < 1:
                add = 0
                while field_a + field_b < 1:
                    add += 1
                field_a += add


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mask_with_offset_written_in_conditional(backend):
    @gtscript.stencil(backend, externals={"mord": 5})
    def stencil(outp: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            cond = True
            if cond[0, -1, 0] or cond[0, 0, 0]:
                outp = 1.0
            else:
                outp = 0.0

    outp = gt_storage.zeros(
        shape=(10, 10, 10), backend=backend, aligned_index=(0, 0, 0), dtype=float
    )

    stencil(outp)

    outp = storage_utils.cpu_copy(outp)
    assert np.allclose(1.0, outp)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_write_data_dim_indirect_addressing(backend):
    INT32_VEC2 = (np.int32, (2,))

    def stencil(
        input_field: gtscript.Field[gtscript.IJK, np.int32],
        output_field: gtscript.Field[gtscript.IJK, INT32_VEC2],
        index: int,
    ):
        with computation(PARALLEL), interval(...):
            output_field[0, 0, 0][index] = input_field

    aligned_index = (0, 0, 0)
    full_shape = (1, 1, 2)
    input_field = gt_storage.ones(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=np.int32
    )
    output_field = gt_storage.zeros(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=INT32_VEC2
    )

    gtscript.stencil(definition=stencil, backend=backend)(input_field, output_field, index := 1)
    assert output_field[0, 0, 0, index] == 1


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_read_data_dim_indirect_addressing(backend):
    INT32_VEC2 = (np.int32, (2,))

    def stencil(
        input_field: gtscript.Field[gtscript.IJK, INT32_VEC2],
        output_field: gtscript.Field[gtscript.IJK, np.int32],
        index: int,
    ):
        with computation(PARALLEL), interval(...):
            output_field = input_field[0, 0, 0][index]

    aligned_index = (0, 0, 0)
    full_shape = (1, 1, 2)
    input_field = gt_storage.ones(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=INT32_VEC2
    )
    output_field = gt_storage.zeros(
        full_shape, backend=backend, aligned_index=aligned_index, dtype=np.int32
    )

    gtscript.stencil(definition=stencil, backend=backend)(input_field, output_field, 1)
    assert output_field[0, 0, 0] == 1


@pytest.mark.parametrize("backend", ALL_BACKENDS)
class TestNegativeOrigin:
    def test_negative_origin_i(self, backend):
        @gtscript.stencil(backend=backend)
        def stencil_i(
            input_field: gtscript.Field[gtscript.IJK, np.int32],
            output_field: gtscript.Field[gtscript.IJK, np.int32],
        ):
            with computation(PARALLEL), interval(...):
                output_field = input_field[1, 0, 0]

        input_field = gt_storage.ones(
            backend=backend, aligned_index=(0, 0, 0), shape=(1, 1, 1), dtype=np.int32
        )
        output_field = gt_storage.zeros(
            backend=backend, aligned_index=(0, 0, 0), shape=(1, 1, 1), dtype=np.int32
        )

        stencil_i(input_field, output_field, origin={"input_field": (-1, 0, 0)})
        assert output_field[0, 0, 0] == 1

    def test_negative_origin_k(self, backend):
        @gtscript.stencil(backend=backend)
        def stencil_k(
            input_field: gtscript.Field[gtscript.IJK, np.int32],
            output_field: gtscript.Field[gtscript.IJK, np.int32],
        ):
            with computation(PARALLEL), interval(...):
                output_field = input_field[0, 0, 1]

        input_field = gt_storage.ones(
            backend=backend, aligned_index=(0, 0, 0), shape=(1, 1, 1), dtype=np.int32
        )
        output_field = gt_storage.zeros(
            backend=backend, aligned_index=(0, 0, 0), shape=(1, 1, 1), dtype=np.int32
        )

        stencil_k(input_field, output_field, origin={"input_field": (0, 0, -1)})
        assert output_field[0, 0, 0] == 1


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_origin_k_fields(backend):
    @gtscript.stencil(backend=backend, rebuild=True)
    def k_to_ijk(outp: Field[np.float64], inp: Field[gtscript.K, np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp

    origin = {"outp": (0, 0, 1), "inp": (2,)}
    domain = (2, 2, 8)

    data = np.arange(10, dtype=np.float64)
    inp = gt_storage.from_array(
        data=data, aligned_index=(0,), dtype=np.float64, dimensions="K", backend=backend
    )
    outp = gt_storage.zeros(
        shape=(2, 2, 10), aligned_index=(0, 0, 0), dtype=np.float64, backend=backend
    )

    k_to_ijk(outp, inp, origin=origin, domain=domain)

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)
    np.testing.assert_allclose(data, inp)
    np.testing.assert_allclose(np.broadcast_to(data[2:], shape=(2, 2, 8)), outp[:, :, 1:-1])
    np.testing.assert_allclose(0.0, outp[:, :, 0])
    np.testing.assert_allclose(0.0, outp[:, :, -1])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_pruned_args_match(backend):
    @gtscript.stencil(backend=backend)
    def test(out: Field[np.float64], inp: Field[np.float64]):
        with computation(PARALLEL), interval(...):
            out = 0.0
            with horizontal(region[I[0] - 1, J[0] - 1]):
                out = inp

    inp = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=(2, 2, 2), dtype=np.float64
    )
    out = gt_storage.empty(
        backend=backend, aligned_index=(0, 0, 0), shape=(2, 2, 2), dtype=np.float64
    )
    test(out, inp)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_direct_datadims_index(backend):
    F64_VEC4 = (np.float64, (2, 2, 2, 2))

    @gtscript.stencil(backend=backend)
    def test(out: Field[np.float64], inp: GlobalTable[F64_VEC4]):
        with computation(PARALLEL), interval(...):
            out = inp.A[1, 0, 1, 0]

    inp = gt_storage.ones(backend=backend, shape=(2, 2, 2, 2), dtype=np.float64)
    inp[1, 0, 1, 0] = 42
    out = gt_storage.zeros(backend=backend, shape=(2, 2, 2), dtype=np.float64)
    test(out, inp)
    assert (out[:] == 42).all()
