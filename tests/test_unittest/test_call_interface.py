# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import gt4py.backend as gt_backend
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import Field, K

from ..definitions import INTERNAL_BACKENDS, INTERNAL_CPU_BACKENDS


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

    A = gt_storage.ones(backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )

    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
    stencil(A, B, C, param=3.0, origin={"_all_": (1, 1, 1), "field1": (2, 2, 2)}, domain=(1, 1, 1))

    assert A[2, 2, 2] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
    stencil(A, B, C, param=3.0, origin={"field1": (2, 2, 2)}, domain=(1, 1, 1))

    assert A[2, 2, 2] == 4
    assert B[2, 2, 2] == 7
    assert C[0, 1, 0] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47


@pytest.mark.parametrize("backend", INTERNAL_BACKENDS)
def test_origin_k_fields(backend):
    @gtscript.stencil(backend=backend, rebuild=True)
    def k_to_ijk(outp: Field[np.float64], inp: Field[gtscript.K, np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp

    origin = {"outp": (0, 0, 1), "inp": (2,)}
    domain = (2, 2, 8)

    data = np.arange(10, dtype=np.float64)
    inp = gt_storage.from_array(
        data=data,
        shape=(10,),
        default_origin=(0,),
        dtype=np.float64,
        mask=[False, False, True],
        backend=backend,
    )
    outp = gt_storage.zeros(
        shape=(2, 2, 10), default_origin=(0, 0, 0), dtype=np.float64, backend=backend
    )

    k_to_ijk(outp, inp, origin=origin, domain=domain)

    inp.device_to_host()
    outp.device_to_host()
    np.testing.assert_allclose(data, np.asarray(inp))
    np.testing.assert_allclose(
        np.broadcast_to(data[2:], shape=(2, 2, 8)), np.asarray(outp)[:, :, 1:-1]
    )
    np.testing.assert_allclose(0.0, np.asarray(outp)[:, :, 0])
    np.testing.assert_allclose(0.0, np.asarray(outp)[:, :, -1])


def test_domain_selection():
    stencil = gtscript.stencil(definition=base_stencil, backend="numpy")

    A = gt_storage.ones(backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )

    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0))
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
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


@pytest.mark.parametrize("backend", INTERNAL_CPU_BACKENDS)
def test_default_arguments(backend):
    branch_true = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": True}, rebuild=True
    )
    branch_false = gtscript.stencil(
        backend=backend, definition=a_stencil, externals={"BRANCH": False}, rebuild=True
    )

    arg1 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
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
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg2 = gt_storage.zeros(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    arg3 = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
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


@pytest.mark.parametrize("backend", INTERNAL_CPU_BACKENDS)
def test_halo_checks(backend):
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test default works
    in_field = gt_storage.ones(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field)
    assert (out_field[1:-1, 1:-1, :] == 1).all()

    # test setting arbitrary, small domain works
    in_field = gt_storage.ones(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(10, 10, 10))
    assert (out_field[2:12, 2:12, :] == 1).all()

    # test setting domain+origin too large raises
    in_field = gt_storage.ones(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(22, 22, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    with pytest.raises(ValueError):
        stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))

    # test 2*origin+domain does not raise if still fits (c.f. previous bug in c++ check.)
    in_field = gt_storage.ones(
        backend=backend, shape=(23, 23, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(23, 23, 10), default_origin=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))


def test_np_int_types():
    backend = "numpy"
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        default_origin=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        default_origin=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    stencil(
        in_field=in_field,
        out_field=out_field,
        origin=(np.int8(2), np.int16(2), np.int32(0)),
        domain=(np.int64(20), int(20), 10),
    )


def test_np_array_int_types():
    backend = "numpy"
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=np.asarray((23, 23, 10), dtype=np.int64),
        default_origin=np.asarray((1, 1, 0), dtype=np.int64),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend,
        shape=np.asarray((23, 23, 10), dtype=np.int64),
        default_origin=np.asarray((1, 1, 0), dtype=np.int64),
        dtype=np.float64,
    )
    stencil(
        in_field=in_field,
        out_field=out_field,
        origin=np.asarray((2, 2, 0), dtype=np.int64),
        domain=np.asarray((20, 20, 10), dtype=np.int64),
    )


def test_ndarray_warning():
    """test that proper warnings are raised depending on field type."""
    backend = "numpy"
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=np.asarray((23, 23, 10), dtype=np.int64),
        default_origin=np.asarray((1, 1, 0), dtype=np.int64),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend,
        shape=np.asarray((23, 23, 10), dtype=np.int64),
        default_origin=np.asarray((1, 1, 0), dtype=np.int64),
        dtype=np.float64,
    )
    with pytest.warns(RuntimeWarning):
        stencil(
            in_field=in_field.view(np.ndarray),
            out_field=out_field.view(np.ndarray),
            origin=np.asarray((2, 2, 0), dtype=np.int64),
            domain=np.asarray((20, 20, 10), dtype=np.int64),
        )

    with pytest.warns(None) as record:
        stencil(
            in_field=in_field,
            out_field=out_field,
            origin=np.asarray((2, 2, 0), dtype=np.int64),
            domain=np.asarray((20, 20, 10), dtype=np.int64),
        )
    assert len(record) == 0


@pytest.mark.parametrize("backend", ["debug", "numpy", "gtx86"])
def test_exec_info(backend):
    """test that proper warnings are raised depending on field type."""
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    exec_info = {}
    # test numpy int types are accepted
    in_field = gt_storage.ones(
        backend=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        default_origin=(1, 1, 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        backend=backend, shape=(23, 23, 10), default_origin=(1, 1, 0), dtype=np.float64
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
    if backend.startswith("gt"):
        assert "run_cpp_start_time" in exec_info
        assert "run_cpp_end_time" in exec_info
        assert exec_info["run_cpp_end_time"] > exec_info["run_cpp_start_time"]


class TestAxesMismatch:
    @pytest.fixture
    def sample_stencil(self):
        @gtscript.stencil(backend="debug")
        def _stencil(
            field_out: gtscript.Field[gtscript.IJ, np.float64],
        ):
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
            match="Storage for '.*' has domain mask '.*' but the API signature expects '\[I, J\]'",
        ):
            sample_stencil(
                field_out=gt_storage.empty(
                    shape=(3, 3),
                    mask=[True, False, True],
                    dtype=np.float64,
                    backend="debug",
                    default_origin=(0, 0),
                )
            )


@pytest.mark.parametrize("backend", INTERNAL_BACKENDS)
def test_origin_unchanged(backend):
    @gtscript.stencil(backend=backend)
    def calc_damp(outp: Field[float], inp: Field[K, float]):
        with computation(FORWARD), interval(...):
            outp = inp

    outp = gt_storage.ones(
        backend=backend,
        default_origin=(1, 1, 1),
        shape=(4, 4, 4),
        dtype=float,
        mask=[True, True, True],
    )
    inp = gt_storage.ones(
        backend=backend,
        default_origin=(1,),
        shape=(4, 4, 4),
        dtype=float,
        mask=[False, False, True],
    )

    origin = {"_all_": (1, 1, 1), "inp": (1,)}
    origin_ref = dict(origin)

    calc_damp(outp, inp, origin=origin, domain=(3, 3, 3))

    assert all(k in origin_ref for k in origin.keys())
    assert all(k in origin for k in origin_ref.keys())
    assert all(v is origin_ref[k] for k, v in origin.items())
