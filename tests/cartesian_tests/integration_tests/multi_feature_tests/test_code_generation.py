# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    K,
    IJ,
    computation,
    horizontal,
    interval,
    region,
    sin,
)
from gt4py.storage.cartesian import utils as storage_utils

from cartesian_tests.definitions import ALL_BACKENDS, CPU_BACKENDS, get_array_library
from cartesian_tests.integration_tests.multi_feature_tests.stencil_definitions import (
    EXTERNALS_REGISTRY as externals_registry,
    REGISTRY as stencil_definitions,
)


rng = np.random.default_rng(1337)


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
                shape=(23,) * len(v.axes),
                aligned_index=(10,) * len(v.axes),
            )
        else:
            args[k] = v(1.5)
    # vertical domain size > 16 required for test_large_k_interval
    stencil(**args, origin=(10, 10, 5), domain=(3, 3, 17))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lazy_stencil(backend):
    @gtscript.lazy_stencil(backend=backend)
    def definition(field_a: Field[np.float64], field_b: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            field_a[0, 0, 0] = field_b


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_temporary_field_declared_in_if(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            if field_a < 0:
                field_b = -field_a
            else:
                field_b = field_a
            field_a = field_b


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_stage_without_effect(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            field_c = 0.0  # noqa: F841


def test_ignore_np_errstate():
    def setup_and_run(backend, **kwargs):
        field_a = gt_storage.zeros(
            dtype=np.float64, backend=backend, shape=(3, 3, 1), aligned_index=(0, 0, 0)
        )

        @gtscript.stencil(backend=backend, **kwargs)
        def divide_by_zero(field_a: Field[np.float64]):  # type: ignore
            with computation(PARALLEL), interval(...):
                field_a = 1.0 / field_a

        divide_by_zero(field_a)

    # Usual behavior: with the numpy backend there is no error
    setup_and_run(backend="numpy")

    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="numpy", ignore_np_errstate=False)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stencil_without_effect(backend):
    def definition1(field_in: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            tmp = 0.0  # noqa: F841

    def definition2(f_in: Field[np.float64]):  # type: ignore
        from __externals__ import flag  # type: ignore

        with computation(PARALLEL), interval(...):
            if __INLINED(flag):
                B = f_in  # noqa: F841

    stencil1 = gtscript.stencil(backend, definition1)
    stencil2 = gtscript.stencil(backend, definition2, externals={"flag": False})

    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )

    # test with explicit domain specified
    stencil1(field_in, domain=(3, 3, 3))
    stencil2(field_in, domain=(3, 3, 3))

    # test without domain specified
    stencil1(field_in)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_merger_induced_interval_block_reordering(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: Field[np.float64], field_out: Field[np.float64]):  # type: ignore
        with computation(BACKWARD):
            with interval(-2, -1):  # block 1
                field_out = field_in
            with interval(0, -2):  # block 2
                field_out = field_in
        with computation(BACKWARD):
            with interval(-1, None):  # block 3
                field_out = 2 * field_in
            with interval(0, -1):  # block 4
                field_out[0, 0, 0] = 3 * field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0:-1], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, -1], 2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_inputs(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        field_3d: Field[gtscript.IJK, np.float64],  # type: ignore
        field_2d: Field[gtscript.IJ, np.float64],  # type: ignore
        field_1d: Field[gtscript.K, np.float64],  # type: ignore
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
                field_3d[0, 0, 0] = tmp[-1, 0, 0]

    full_shape = (6, 6, 6)
    aligned_index = (1, 1, 0)
    dtype = float

    field_3d = gt_storage.zeros(
        full_shape, dtype, backend=backend, aligned_index=aligned_index, dimensions=None
    )
    assert field_3d.shape == full_shape[:], "field_3d shape"

    field_2d = gt_storage.zeros(
        full_shape[:-1],
        dtype,
        backend=backend,
        aligned_index=aligned_index[:-1],
        dimensions="IJ",
    )
    assert field_2d.shape == full_shape[:-1], "field_2d shape"

    field_1d = gt_storage.ones(
        full_shape[-1:],
        dtype,
        backend=backend,
        aligned_index=(aligned_index[-1],),
        dimensions="K",
    )
    assert list(field_1d.shape) == [full_shape[-1]], "field_1d shape"

    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0), domain=(4, 3, 6))
    res_field_3d = storage_utils.cpu_copy(field_3d)
    np.testing.assert_allclose(res_field_3d[1:-1, 1:-2, :1], 2, err_msg="expected 2 from K=0")
    np.testing.assert_allclose(res_field_3d[1:-1, 1:-2, 1:], 1, err_msg="expected 1 from K>=1")
    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_masked(backend):
    @gtscript.stencil(backend=backend)
    def copy_2to3(
        cond: Field[gtscript.IJK, np.float64],  # type: ignore
        inp: Field[gtscript.IJ, np.float64],  # type: ignore
        outp: Field[gtscript.IJK, np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            if cond > 0.0:
                outp[0, 0, 0] = inp

    inp = rng.standard_normal((10, 10))
    outp = rng.standard_normal((10, 10, 10))
    cond = rng.standard_normal((10, 10, 10))

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
        cond: Field[gtscript.IJK, np.float64],  # type: ignore
        inp: Field[gtscript.IJ, np.float64],  # type: ignore
        outp: Field[gtscript.IJK, np.float64],  # type: ignore
    ):
        with computation(FORWARD), interval(...):
            if cond > 0.0:
                outp[0, 0, 0] = inp

    inp = rng.standard_normal((10, 10))
    outp = rng.standard_normal((10, 10, 10))
    cond = rng.standard_normal((10, 10, 10))

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
        inp: Field[gtscript.IJ, np.float64],  # type: ignore
        outp: Field[gtscript.IJK, np.float64],  # type: ignore
    ):
        with computation(FORWARD), interval(...):
            outp[0, 0, 0] = inp

    inp_f = gt_storage.from_array(
        rng.standard_normal((10, 10)), aligned_index=(0, 0), backend=backend
    )
    outp_f = gt_storage.from_array(
        rng.standard_normal((10, 10, 10)), aligned_index=(0, 0, 0), backend=backend
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
        field: Field[np.float64],  # type: ignore
        vec_field: Field[FLOAT64_VEC2],  # type: ignore
        mat_field: Field[FLOAT64_MAT22],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            tmp = vec_field[0, 0, 0][0] + vec_field[0, 0, 0][1]  # noqa: F841

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
        in_field: Field[np.float64],  # type: ignore
        parameter: np.float64,
        out_field: Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field * parameter

    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(23, 23, 23), aligned_index=(0, 0, 0)
    )

    stencil(field_in, 3.1415, field_out)

    np.testing.assert_allclose(field_out[:, :, :], 3.1415)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_variable_offsets(backend):
    @gtscript.stencil(backend=backend)
    def stencil_ij(
        in_field: Field[np.float64],  # type: ignore
        out_field: Field[np.float64],  # type: ignore
        index_field: Field[gtscript.IJ, int],  # type: ignore
    ):
        with computation(FORWARD), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 1] + in_field[0, 0, index_field + 1]
            index_field = index_field + 1

    @gtscript.stencil(backend=backend)
    def stencil_ijk(
        in_field: Field[np.float64],  # type: ignore
        out_field: Field[np.float64],  # type: ignore
        index_field: Field[int],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 1] + in_field[0, 0, index_field + 1]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_variable_offsets_and_while_loop(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        pe1: Field[np.float64],  # type: ignore
        pe2: Field[np.float64],  # type: ignore
        qin: Field[np.float64],  # type: ignore
        qout: Field[np.float64],  # type: ignore
        lev: Field[gtscript.IJ, np.int_],  # type: ignore
    ):
        with computation(FORWARD), interval(0, -1):
            if pe2[0, 0, 1] <= pe1[0, 0, lev]:
                qout = qin[0, 0, 1]
            else:
                qsum = pe1[0, 0, lev + 1] - pe2[0, 0, lev]
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += qin[0, 0, lev] / (pe2[0, 0, 1] - pe1[0, 0, lev])
                    lev = lev + 1
                qout[0, 0, 0] = qsum / (pe2[0, 0, 1] - pe2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_nested_while_loop(backend):
    @gtscript.stencil(backend=backend)
    def stencil(field_a: Field[np.float64], field_b: Field[np.int_]):  # type: ignore
        with computation(PARALLEL), interval(...):
            while field_a < 1:
                add = 0
                while field_a + field_b < 1:
                    add += 1
                field_a += add


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mask_with_offset_written_in_conditional(backend):
    @gtscript.stencil(backend)
    def stencil(outp: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            cond = True
            if cond[0, -1, 0] or cond[0, 0, 0]:
                outp = 1.0
            else:
                outp[0, 0, 0] = 0.0

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
        input_field: Field[gtscript.IJK, np.int32],  # type: ignore
        output_field: Field[gtscript.IJK, INT32_VEC2],  # type: ignore
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
        input_field: Field[gtscript.IJK, INT32_VEC2],  # type: ignore
        output_field: Field[gtscript.IJK, np.int32],  # type: ignore
        index: int,
    ):
        with computation(PARALLEL), interval(...):
            output_field[0, 0, 0] = input_field[0, 0, 0][index]

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
            input_field: Field[gtscript.IJK, np.int32],  # type: ignore
            output_field: Field[gtscript.IJK, np.int32],  # type: ignore
        ):
            with computation(PARALLEL), interval(...):
                output_field[0, 0, 0] = input_field[1, 0, 0]

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
            input_field: Field[gtscript.IJK, np.int32],  # type: ignore
            output_field: Field[gtscript.IJK, np.int32],  # type: ignore
        ):
            with computation(PARALLEL), interval(...):
                output_field[0, 0, 0] = input_field[0, 0, 1]

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
    def k_to_ijk(outp: Field[np.float64], inp: Field[gtscript.K, np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            outp[0, 0, 0] = inp

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
def test_tmp_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):  # type: ignore
        with computation(PARALLEL):
            with interval(...):
                tmp = field_in + 1
        with computation(PARALLEL):
            with interval(...):
                field_out[0, 0, 0] = tmp[-1, 0, 0] + tmp[1, 0, 0]

    stencil(field_in, field_out, origin=(1, 1, 0), domain=(4, 4, 6))

    # the inside of the domain is 4
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[1:-1, 1:-1, :], 4)
    # the rest is 0
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[0:1, :, :], 0)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[-1:, :, :], 0)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, 0:1, :], 0)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, -1:, :], 0)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backward_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):  # type: ignore
        with computation(BACKWARD):
            with interval(-1, None):
                field_in = 2
                field_out = field_in
            with interval(0, -1):
                field_in = field_in[0, 0, 1] + 1
                field_out[0, 0, 0] = field_in

    stencil(field_in, field_out)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 0], 5)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 1], 4)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 2], 3)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 3], 2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_while_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):  # type: ignore
        with computation(PARALLEL):
            with interval(...):
                while field_in < 10:
                    field_in += 1
                field_out[0, 0, 0] = field_in

    stencil(field_in, field_out)

    # the inside of the domain is 10
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, :], 10)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_higher_dim_literal_stencil(backend):
    FLOAT64_NDDIM = (np.float64, (4,))

    field_in = gt_storage.ones(
        dtype=FLOAT64_NDDIM, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_in[:, :, :, 2] = 5

    @gtscript.stencil(backend=backend)
    def stencil(
        vec_field: gtscript.Field[FLOAT64_NDDIM],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = vec_field[0, 0, 0][2]

    stencil(field_in, field_out)

    # the inside of the domain is 5
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, :], 5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_higher_dim_scalar_stencil(backend):
    FLOAT64_NDDIM = (np.float64, (4,))

    field_in = gt_storage.ones(
        dtype=FLOAT64_NDDIM, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_in[:, :, :, 2] = 5

    @gtscript.stencil(backend=backend)
    def stencil(
        vec_field: gtscript.Field[FLOAT64_NDDIM],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
        scalar_argument: int,
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = vec_field[0, 0, 0][scalar_argument]

    stencil(field_in, field_out, 2)

    # the inside of the domain is 5
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, :], 5)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_native_function_call_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 0] + sin(0.848062)

    test_stencil(field_in, field_out)
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, :], 1.75)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_unary_operator_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = -in_field[0, 0, 0]

    test_stencil(field_in, field_out)
    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, :], -1)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_ternary_operator_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[0, 0, 1] = 20

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, 0] if in_field > 10 else in_field[0, 0, 0] + 1

    test_stencil(field_in, field_out)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[0, 0, 1], 20)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[1:, 1:, 1], 2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_mask_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[0, 0, 1] = -20

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            if in_field[0, 0, 0] > 0:
                out_field[0, 0, 0] = in_field
            else:
                out_field[0, 0, 0] = 1

    test_stencil(field_in, field_out)

    cpu_output = storage_utils.cpu_copy(field_out)
    assert np.all(cpu_output.view(np.ndarray) > 0)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_k_offset_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:, :, 0] *= 10
    offset = -1

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
        scalar_value: int,
    ):
        with computation(PARALLEL), interval(1, None):
            out_field[0, 0, 0] = in_field[0, 0, scalar_value]

    test_stencil(field_in, field_out, offset)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 1], 10)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_k_offset_field_stencil(backend):
    field_in = gt_storage.ones(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_idx = gt_storage.ones(dtype=np.int64, backend=backend, shape=(4, 4), aligned_index=(0, 0))
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )
    field_in[:, :, 0] *= 10
    field_idx[:, :] *= -2

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
        idx_field: gtscript.Field[gtscript.IJ, np.int64],  # type: ignore
    ):
        with computation(PARALLEL), interval(1, None):
            out_field[0, 0, 0] = in_field[0, 0, idx_field + 1]

    test_stencil(field_in, field_out, field_idx)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[:, :, 1], 10)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_k_only_access_stencil(backend):
    field_in = gt_storage.from_array(
        np.array([2, 3, 4, 5]), dtype=np.float64, backend=backend, aligned_index=(0,)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def test_stencil(
        in_field: gtscript.Field[gtscript.K, np.float64],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL):
            with interval(0, 1):
                out_field[0, 0, 0] = in_field[1]
            with interval(1, None):
                out_field[0, 0, 0] = in_field[-1]

    test_stencil(field_in, field_out)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[1, 1, :], [3, 2, 3, 4])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_table_access_stencil(backend):
    table_view = gt_storage.from_array(
        np.array([2, 3, 4, 5]), dtype=np.float64, backend=backend, aligned_index=(0,)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend=backend, shape=(4, 4, 4), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def test_stencil(
        table_view: gtscript.GlobalTable[(np.float64, (4))],  # type: ignore
        out_field: gtscript.Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL):
            with interval(0, 1):
                out_field[0, 0, 0] = table_view.A[1]
            with interval(1, None):
                out_field[0, 0, 0] = table_view.A[2]

    test_stencil(table_view, field_out)

    cpu_output = storage_utils.cpu_copy(field_out)
    np.testing.assert_allclose(cpu_output.view(np.ndarray)[1, 1, :], [3, 4, 4, 4])


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_pruned_args_match(backend):
    @gtscript.stencil(backend=backend)
    def test(out: Field[np.float64], inp: Field[np.float64]):  # type: ignore
        with computation(PARALLEL), interval(...):
            out = 0.0
            with horizontal(region[I[0] - 1, J[0] - 1]):
                out[0, 0, 0] = inp

    inp = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=(2, 2, 2), dtype=np.float64
    )
    out = gt_storage.empty(
        backend=backend, aligned_index=(0, 0, 0), shape=(2, 2, 2), dtype=np.float64
    )
    test(out, inp)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_K_offset_write(backend):
    # Cuda generates bad code for the K offset
    if backend == "cuda":
        pytest.skip("cuda K-offset write generates bad code")

    if backend in ["gt:gpu", "dace:gpu"]:
        pytest.skip(
            f"{backend} backend is not capable of K offset write, bug remains unsolved: https://github.com/GridTools/gt4py/issues/1684"
        )

    arraylib = get_array_library(backend)
    array_shape = (1, 1, 4)
    K_values = arraylib.arange(start=40, stop=44)

    # Simple case of writing ot an offset.
    # A is untouched
    # B is written in K+1 and should have K_values, except for the first element (FORWARD)
    @gtscript.stencil(backend=backend)
    def simple(A: Field[np.float64], B: Field[np.float64]):  # type: ignore
        with computation(FORWARD), interval(...):
            B[0, 0, 1] = A

    A = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    A[:, :, :] = K_values
    B = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    simple(A, B)
    assert (B[:, :, 0] == 0).all()
    assert (B[:, :, 1:3] == K_values[0:2]).all()

    # Order of operations: FORWARD with negative offset
    # means while A is update B will have non-updated values of A
    # Because of the interval, value of B[0] is 0
    @gtscript.stencil(backend=backend)
    def forward(A: Field[np.float64], B: Field[np.float64], scalar: np.float64):  # type: ignore
        with computation(FORWARD), interval(1, None):
            A[0, 0, -1] = scalar
            B[0, 0, 0] = A

    A = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    A[:, :, :] = K_values
    B = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    forward(A, B, 2.0)
    assert (A[:, :, :3] == 2.0).all()
    assert (A[:, :, 3] == K_values[3]).all()
    assert (B[:, :, 0] == 0).all()
    assert (B[:, :, 1:] == K_values[1:]).all()

    # Order of operations: BACKWARD with negative offset
    # means A is update B will get the updated values of A
    @gtscript.stencil(backend=backend)
    def backward(A: Field[np.float64], B: Field[np.float64], scalar: np.float64):  # type: ignore
        with computation(BACKWARD), interval(1, None):
            A[0, 0, -1] = scalar
            B[0, 0, 0] = A

    A = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    A[:, :, :] = K_values
    B = gt_storage.empty(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    backward(A, B, 2.0)
    assert (A[:, :, :3] == 2.0).all()
    assert (A[:, :, 3] == K_values[3]).all()
    assert (B[:, :, :] == A[:, :, :]).all()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_K_offset_write_conditional(backend):
    if backend == "cuda":
        pytest.skip("Cuda backend is not capable of K offset write")

    arraylib = get_array_library(backend)
    array_shape = (1, 1, 4)
    K_values = arraylib.arange(start=40, stop=44)

    @gtscript.stencil(backend=backend)
    def column_physics_conditional(A: Field[np.float64], B: Field[np.float64], scalar: np.float64):  # type: ignore
        with computation(BACKWARD), interval(1, -1):
            if A > 0 and B > 0:
                A[0, 0, -1] = scalar
                B[0, 0, 1] = A
            lev = 1
            while A >= 0 and B >= 0:
                A[0, 0, lev] = -1
                B = -1
                lev = lev + 1

    A = gt_storage.zeros(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    A[:, :, :] = K_values
    B = gt_storage.ones(
        backend=backend, aligned_index=(0, 0, 0), shape=array_shape, dtype=np.float64
    )
    column_physics_conditional(A, B, 2.0)
    # Manual unroll of the above
    # Starts with
    # - A[...] = [40, 41, 42, 43]
    # - B[...] = [1, 1, 1, 1]
    # Now in-stencil
    # ITERATION k = 2 of [2:1]
    # if condition
    # - A[2] == 42 && B[2] == 1 => True
    # - A[1] = 2.0
    # - B[3] = A[2] = 42
    # while
    # - lev = 1
    # - A[2] == 42 && B[2] == 1 => True
    # - A[3] = -1
    # - B[2] = -1
    # - lev = 2
    # - A[2] == 42 && B[2] == -1 => False
    # End of iteration state
    # - A[...] = A[40, 2.0, 42, -1]
    # - B[...] = B[1, 1, -1, 42]
    # ITERATION k = 1 of [2:1]
    # if condition
    # - A[1] == 2.0 && B[1] == 1 => True
    # - A[0] = 2.0
    # - B[2] = A[1] = 2.0
    # while
    # - lev = 1
    # - A[1] == 2.0 && B[1] == 1 => True
    # - A[2] = -1
    # - B[1] = -1
    # - lev = 2
    # - A[1] == 2.0 && B[1] == -1 => False
    # End of stencil state
    # - A[...] = A[2.0, 2.0, -1, -1]
    # - B[...] = B[1, -1, 2.0, 42]

    assert (A[0, 0, :] == arraylib.array([2, 2, -1, -1])).all()
    assert (B[0, 0, :] == arraylib.array([1, -1, 2, 42])).all()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_direct_datadims_index(backend):
    F64_VEC4 = (np.float64, (2, 2, 2, 2))

    @gtscript.stencil(backend=backend)
    def test(out: Field[np.float64], inp: GlobalTable[F64_VEC4]):  # type: ignore
        with computation(PARALLEL), interval(...):
            out[0, 0, 0] = inp.A[1, 0, 1, 0]

    inp = gt_storage.ones(backend=backend, shape=(2, 2, 2, 2), dtype=np.float64)
    inp[1, 0, 1, 0] = 42
    out = gt_storage.zeros(backend=backend, shape=(2, 2, 2), dtype=np.float64)
    test(out, inp)
    assert (out[:] == 42).all()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_function_inline_in_while(backend):
    @gtscript.function
    def add_42(v):
        return v + 42

    @gtscript.stencil(backend=backend)
    def test(
        in_field: Field[np.float64],  # type: ignore
        out_field: Field[np.float64],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            count = 1
            while count < 10:
                sa = add_42(out_field)
                out_field = in_field + sa
                count = count + 1

    domain = (5, 5, 2)
    in_arr = gt_storage.ones(backend=backend, shape=domain, dtype=np.float64)
    out_arr = gt_storage.ones(backend=backend, shape=domain, dtype=np.float64)
    test(in_arr, out_arr)
    assert (out_arr[:, :, :] == 388.0).all()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_cast_in_index(backend):
    @gtscript.stencil(backend)
    def cast_in_index(
        in_field: Field[np.float64],  # type: ignore
        i32: np.int32,
        i64: np.int64,
        out_field: Field[np.float64],  # type: ignore
    ):
        """Simple copy stencil with forced cast in index calculation."""
        with computation(PARALLEL), interval(...):
            out_field[0, 0, 0] = in_field[0, 0, i32 - i64]


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_read_after_write_stencil(backend):
    """Stencil with multiple read after write access patterns."""

    @gtscript.stencil(backend=backend)
    def lagrangian_contributions(
        q: Field[np.float64],  # type: ignore
        pe1: Field[np.float64],  # type: ignore
        pe2: Field[np.float64],  # type: ignore
        q4_1: Field[np.float64],  # type: ignore
        q4_2: Field[np.float64],  # type: ignore
        q4_3: Field[np.float64],  # type: ignore
        q4_4: Field[np.float64],  # type: ignore
        dp1: Field[np.float64],  # type: ignore
        lev: Field[gtscript.IJ, np.int64],  # type: ignore
    ):
        """
        Args:
            q (out):
            pe1 (in):
            pe2 (in):
            q4_1 (in):
            q4_2 (in):
            q4_3 (in):
            q4_4 (in):
            dp1 (in):
            lev (inout):
        """
        with computation(FORWARD), interval(...):
            pl = (pe2 - pe1[0, 0, lev]) / dp1[0, 0, lev]
            if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
                pr = (pe2[0, 0, 1] - pe1[0, 0, lev]) / dp1[0, 0, lev]
                q[0, 0, 0] = (
                    q4_2[0, 0, lev]
                    + 0.5 * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev]) * (pr + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
                )
            else:
                qsum = (pe1[0, 0, lev + 1] - pe2) * (
                    q4_2[0, 0, lev]
                    + 0.5 * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev]) * (1.0 + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
                )
                lev = lev + 1
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                    lev = lev + 1
                dp = pe2[0, 0, 1] - pe1[0, 0, lev]
                esl = dp / dp1[0, 0, lev]
                qsum += dp * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * esl
                    * (
                        q4_3[0, 0, lev]
                        - q4_2[0, 0, lev]
                        + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                    )
                )
                q = qsum / (pe2[0, 0, 1] - pe2)
            lev = lev - 1


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            backend,
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason="Absolute K indexing not yet supported."
            ),
        )
        for backend in ["gt:cpu_ifirst", "numpy"]
    ],
)
def test_absolute_K_index_raise(backend):
    @gtscript.stencil(backend=backend)
    def test_absolute_k_index(in_field: Field[np.float64], out_field: Field[np.float64]) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field.at(K=2)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug"),
        pytest.param("dace:cpu", marks=[pytest.mark.requires_dace]),
        pytest.param("dace:gpu", marks=[pytest.mark.requires_dace, pytest.mark.requires_gpu]),
    ],
)
def test_absolute_K_index(backend):
    domain = (5, 5, 5)

    in_arr = gt_storage.ones(backend=backend, shape=domain, dtype=np.float64)
    idx_arr = gt_storage.zeros(backend=backend, shape=(domain[0], domain[1]), dtype=np.int64)
    k_arr = gt_storage.zeros(backend=backend, shape=(domain[2],), dtype=np.float64)
    out_arr = gt_storage.zeros(backend=backend, shape=domain, dtype=np.float64)

    @gtscript.stencil(backend=backend)
    def test_literal_access(in_field: Field[np.float64], out_field: Field[np.float64]) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field.at(K=2)

    in_arr[:, :, :] = 1
    in_arr[:, :, 2] = 42.42
    out_arr[:, :, :] = 0
    test_literal_access(in_arr, out_arr)
    assert (out_arr[:, :, :] == 42.42).all()

    @gtscript.stencil(backend=backend)
    def test_parameter_access(
        in_field: Field[np.float64], out_field: Field[np.float64], idx: int
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field.at(K=idx)

    in_arr[:, :, :] = 1
    in_arr[:, :, 3] = 42.42
    out_arr[:, :, :] = 0
    test_parameter_access(in_arr, out_arr, 3)
    assert (out_arr[:, :, :] == 42.42).all()

    @gtscript.stencil(backend=backend, externals={"K4": 4})
    def test_external_access(in_field: Field[np.float64], out_field: Field[np.float64]) -> None:
        with computation(PARALLEL), interval(...):
            from __externals__ import K4

            out_field = in_field.at(K=K4)

    in_arr[:, :, :] = 1
    in_arr[:, :, 4] = 42.42
    out_arr[:, :, :] = 0
    test_external_access(in_arr, out_arr)
    assert (out_arr[:, :, :] == 42.42).all()

    @gtscript.stencil(backend=backend)
    def test_field_access(
        in_field: Field[np.float64], index_field: Field[IJ, np.int64], out_field: Field[np.float64]
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field.at(K=index_field)

    in_arr[:, :, :] = 1
    idx_arr[:, :] = 1
    in_arr[:, :, 1] = 42.42
    out_arr[:, :, :] = 0
    test_field_access(in_arr, idx_arr, out_arr)
    assert (out_arr[:, :, :] == 42.42).all()

    @gtscript.stencil(backend=backend)
    def test_lower_dim_field(
        k_field: Field[K, np.float64],
        out_field: Field[np.float64],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = k_field.at(K=2)

    k_arr[:] = 0
    k_arr[2] = 42.42
    out_arr[:, :, :] = 0
    test_lower_dim_field(k_arr, out_arr)
    assert (out_arr[:, :, :] == 42.42).all()
