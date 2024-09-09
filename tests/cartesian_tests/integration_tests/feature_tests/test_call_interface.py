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

import gt4py.cartesian.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.cartesian.gtscript import Field, K

from cartesian_tests.definitions import ALL_BACKENDS, CPU_BACKENDS
from cartesian_tests.utils import DimensionsWrapper, OriginWrapper


def base_stencil(
    field1: Field[np.float64],
    field2: Field[np.float64],
    field3: Field[np.float32],
    *,
    param: np.float64,
):
    with computation(PARALLEL), interval(...):
        field1 = field2 + field3 * param
        field2 = field1 + field3 * param
        field3 = param * field2


def test_origin_selection():
    stencil = gtscript.stencil(definition=base_stencil, backend="numpy")

    A = gt_storage.ones(
        backend="gt:cpu_ifirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    Awrap = OriginWrapper(array=A, origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gt:cpu_kfirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(2, 2, 2)
    )
    Bwrap = OriginWrapper(array=B, origin=(2, 2, 2))
    C = gt_storage.ones(backend="numpy", dtype=np.float32, shape=(3, 3, 3), aligned_index=(0, 1, 0))
    Cwrap = OriginWrapper(array=C, origin=(0, 1, 0))

    stencil(Awrap, Bwrap, Cwrap, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(A) == 30
    assert np.sum(B) == 33
    assert np.sum(C) == 47

    A = gt_storage.ones(
        backend="gt:cpu_ifirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    Awrap = OriginWrapper(array=A, origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gt:cpu_kfirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(2, 2, 2)
    )
    Bwrap = OriginWrapper(array=B, origin=(2, 2, 2))
    C = gt_storage.ones(backend="numpy", dtype=np.float32, shape=(3, 3, 3), aligned_index=(0, 1, 0))
    Cwrap = OriginWrapper(array=C, origin=(0, 1, 0))
    stencil(
        Awrap,
        Bwrap,
        Cwrap,
        param=3.0,
        origin={"_all_": (1, 1, 1), "field1": (2, 2, 2)},
        domain=(1, 1, 1),
    )

    assert A[2, 2, 2] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(A) == 30
    assert np.sum(B) == 33
    assert np.sum(C) == 47

    A = gt_storage.ones(
        backend="gt:cpu_ifirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    Awrap = OriginWrapper(array=A, origin=(0, 0, 0))

    B = gt_storage.ones(
        backend="gt:cpu_kfirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(2, 2, 2)
    )
    Bwrap = OriginWrapper(array=B, origin=(2, 2, 2))
    C = gt_storage.ones(backend="numpy", dtype=np.float32, shape=(3, 3, 3), aligned_index=(0, 1, 0))
    Cwrap = OriginWrapper(array=C, origin=(0, 1, 0))
    stencil(Awrap, Bwrap, Cwrap, param=3.0, origin={"field1": (2, 2, 2)}, domain=(1, 1, 1))

    assert A[2, 2, 2] == 4
    assert B[2, 2, 2] == 7
    assert C[0, 1, 0] == 21
    assert np.sum(A) == 30
    assert np.sum(B) == 33
    assert np.sum(C) == 47


def test_domain_selection():
    stencil = gtscript.stencil(definition=base_stencil, backend="numpy")

    A = gt_storage.ones(
        backend="gt:cpu_ifirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    B = gt_storage.ones(
        backend="gt:cpu_kfirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(2, 2, 2)
    )
    C = gt_storage.ones(backend="numpy", dtype=np.float32, shape=(3, 3, 3), aligned_index=(0, 1, 0))

    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(
        backend="gt:cpu_ifirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    B = gt_storage.ones(
        backend="gt:cpu_kfirst", dtype=np.float64, shape=(3, 3, 3), aligned_index=(2, 2, 2)
    )
    C = gt_storage.ones(backend="numpy", dtype=np.float32, shape=(3, 3, 3), aligned_index=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin=(0, 0, 0))

    assert np.all(A == 4)
    assert np.all(B == 7)
    assert np.all(C == 21)


def a_stencil(
    arg1: Field[np.float64],
    arg2: Field[np.float64],
    arg3: Field[np.float64] = None,
    *,
    par1: np.float64,
    par2: np.float64 = 7.0,
    par3: np.float64 = None,
):
    from __externals__ import BRANCH

    with computation(PARALLEL), interval(...):
        if __INLINED(BRANCH):
            arg1 = arg1 * par1 * par2
        else:
            arg1 = arg2 + arg3 * par1 * par2 * par3


## The following type ignores are there because mypy get's confused by gtscript
def avg_stencil(in_field: Field[np.float64], out_field: Field[np.float64]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        out_field = 0.25 * (
            +in_field[0, 1, 0] + in_field[0, -1, 0] + in_field[1, 0, 0] + in_field[-1, 0, 0]
        )


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_default_arguments(backend):
    branch_true = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": True}, rebuild=True
    )
    branch_false = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": False}, rebuild=True
    )

    arg1 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, None, arg3, par1=2.0)
    np.testing.assert_equal(arg1, 14 * np.ones((3, 3, 3)))
    branch_true(arg1, None, par1=2.0)
    np.testing.assert_equal(arg1, 196 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par3=2.0)
    np.testing.assert_equal(arg1, 56 * np.ones((3, 3, 3)))

    with pytest.raises((ValueError, AssertionError)):
        branch_false(arg1, arg2, par1=2.0, par3=2.0)

    arg1 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), aligned_index=(0, 0, 0)
    )
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, arg2=None, par1=2.0, par2=5.0, par3=3.0)
    np.testing.assert_equal(arg1, 10 * np.ones((3, 3, 3)))
    branch_true(arg1, arg2=None, par1=2.0, par2=5.0)
    np.testing.assert_equal(arg1, 100 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par2=5.0, par3=3.0)
    np.testing.assert_equal(arg1, 60 * np.ones((3, 3, 3)))

    with pytest.raises((TypeError, AssertionError)):
        branch_false(arg1, arg2, arg3, par1=2.0, par2=5.0)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_halo_checks(backend):
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test default works
    in_field = OriginWrapper(
        array=gt_storage.ones(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    out_field = OriginWrapper(
        array=gt_storage.zeros(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    stencil(in_field=in_field, out_field=out_field)
    assert (out_field.array[1:-1, 1:-1, :] == 1).all()

    # test setting arbitrary, small domain works
    in_field = OriginWrapper(
        array=gt_storage.ones(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    out_field = OriginWrapper(
        array=gt_storage.zeros(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(10, 10, 10))
    assert (out_field.array[2:12, 2:12, :] == 1).all()

    # test setting domain+origin too large raises
    in_field = OriginWrapper(
        array=gt_storage.ones(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    out_field = OriginWrapper(
        array=gt_storage.zeros(
            backend=backend, shape=(22, 22, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    with pytest.raises(ValueError):
        stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))

    # test 2*origin+domain does not raise if still fits (c.f. previous bug in c++ check.)
    in_field = OriginWrapper(
        array=gt_storage.ones(
            backend=backend, shape=(23, 23, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    out_field = OriginWrapper(
        array=gt_storage.zeros(
            backend=backend, shape=(23, 23, 10), aligned_index=(1, 1, 0), dtype=np.float64
        ),
        origin=(1, 1, 0),
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))


def test_np_int_types():
    backend = "numpy"
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        aligned_index=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        aligned_index=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    stencil(
        in_field=in_field,
        out_field=out_field,
        origin=(np.int8(2), np.int16(2), np.int32(0)),
        domain=(np.int64(20), int(20), 10),
    )


@pytest.mark.parametrize("backend", ["numpy", "gt:cpu_kfirst"])
def test_exec_info(backend):
    """Test that proper warnings are raised depending on field type."""
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    exec_info = {}
    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        aligned_index=(1, 1, 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(23, 23, 10), aligned_index=(1, 1, 0), dtype=np.float64
    )
    stencil(
        in_field=in_field,
        out_field=out_field,
        origin=(2, 2, 0),
        domain=(20, 20, 10),
        exec_info=exec_info,
    )
    timings = ["call", "call_run", "run"]
    assert all([k + "_start_time" in exec_info for k in timings])
    assert all([k + "_end_time" in exec_info for k in timings])
    assert all([exec_info[k + "_end_time"] > exec_info[k + "_start_time"] for k in timings])
    if backend.startswith("gt:"):
        assert "run_cpp_start_time" in exec_info
        assert "run_cpp_end_time" in exec_info
        assert exec_info["run_cpp_end_time"] > exec_info["run_cpp_start_time"]


class TestAxesMismatch:
    @pytest.fixture
    def sample_stencil(self):
        @gtscript.stencil(backend="numpy")
        def _stencil(field_out: gtscript.Field[gtscript.IJ, np.float64]):
            with computation(FORWARD), interval(...):
                field_out = 1.0

        return _stencil

    def test_ndarray(self, sample_stencil):
        with pytest.raises(
            ValueError, match="Storage for '.*' has 3 dimensions but the API signature expects 2 .*"
        ):
            sample_stencil(field_out=np.ndarray((3, 3, 3), np.float64))

    def test_storage(self, sample_stencil):
        with pytest.raises(
            Exception,
            match="Storage for '.*' has dimensions '.*' but the API signature expects '\[I, J\]'",
        ):
            sample_stencil(
                field_out=DimensionsWrapper(
                    array=gt_storage.empty(
                        shape=(3, 3),
                        dimensions=["I", "K"],
                        dtype=np.float64,
                        backend="numpy",
                        aligned_index=(0, 0),
                    ),
                    dimensions=("I", "K"),
                )
            )


class TestDataDimensions:
    backend = "numpy"

    @pytest.fixture
    def sample_stencil(self):
        @gtscript.stencil(backend=self.backend)
        def _stencil(field_out: gtscript.Field[gtscript.IJK, (np.float64, (2,))]):
            with computation(FORWARD), interval(...):
                field_out[0, 0, 0][0] = 0.0
                field_out[0, 0, 0][1] = 1.0

        return _stencil

    def test_mismatch(self, sample_stencil):
        with pytest.raises(
            ValueError, match="Field '.*' expects data dimensions \(2,\) but got \(3,\)"
        ):
            sample_stencil(
                field_out=gt_storage.empty(
                    shape=(3, 3, 1),
                    dimensions=["I", "J", "K"],
                    dtype=(np.float64, (3,)),
                    backend=self.backend,
                    aligned_index=(0, 0, 0),
                )
            )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_origin_unchanged(backend):
    @gtscript.stencil(backend=backend)
    def calc_damp(outp: Field[float], inp: Field[K, float]):
        with computation(FORWARD), interval(...):
            outp = inp

    outp = gt_storage.ones(
        backend=backend, aligned_index=(1, 1, 1), shape=(4, 4, 4), dtype=float, dimensions="IJK"
    )
    inp = gt_storage.ones(
        backend=backend, aligned_index=(1,), shape=(4,), dtype=float, dimensions="K"
    )

    origin = {"_all_": (1, 1, 1), "inp": (1,)}
    origin_ref = copy.deepcopy(origin)

    calc_damp(outp, inp, origin=origin, domain=(3, 3, 3))

    # NOTE: 2022.10.24 Seems like the StencilObject adds entries for each argument if "_all_" keyword exists.
    # Therefore, this test is changed to only assert that origin is a superset of origin, not strict equality.
    assert all(origin.get(k) == v for k, v in origin_ref.items())


def test_permute_axes():
    @gtscript.stencil(backend="numpy")
    def calc_damp(outp: Field[float], inp: Field[K, float]):
        with computation(FORWARD), interval(...):
            outp = inp

    outp = gt_storage.ones(
        backend="numpy", aligned_index=(1, 1, 1), shape=(4, 4, 4), dtype=float, dimensions="KJI"
    )
    outp_wrap = DimensionsWrapper(array=outp, dimensions="KJI")

    inp = gt_storage.from_array(
        data=np.arange(4), backend="numpy", aligned_index=(1,), dtype=float, dimensions="K"
    )

    calc_damp(outp_wrap, inp)

    for i in range(4):
        np.testing.assert_equal(outp[i, :, :], i)
