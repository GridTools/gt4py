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


try:
    import xarray
except ModuleNotFoundError:
    xarray = None


import gt4py.backend as gt_backend
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import Field

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS


INTERNAL_CPU_BACKENDS = list(set(CPU_BACKENDS) & set(INTERNAL_BACKENDS))
INTERNAL_GPU_BACKENDS = list(set(GPU_BACKENDS) & set(INTERNAL_BACKENDS))


@gtscript.stencil(backend="numpy")
def stencil(
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
    A = gt_storage.ones(defaults="gtmc", dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    B = gt_storage.ones(defaults="gtx86", dtype=np.float64, shape=(3, 3, 3), halo=(2, 2, 2))
    C = gt_storage.ones(defaults="numpy", dtype=np.float32, shape=(3, 3, 3), halo=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(defaults="gtmc", dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    B = gt_storage.ones(defaults="gtx86", dtype=np.float64, shape=(3, 3, 3), halo=(2, 2, 2))
    C = gt_storage.ones(defaults="numpy", dtype=np.float32, shape=(3, 3, 3), halo=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin={"_all_": (1, 1, 1), "field1": (2, 2, 2)}, domain=(1, 1, 1))

    assert A[2, 2, 2] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(defaults="gtmc", dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    B = gt_storage.ones(defaults="gtx86", dtype=np.float64, shape=(3, 3, 3), halo=(2, 2, 2))
    C = gt_storage.ones(defaults="numpy", dtype=np.float32, shape=(3, 3, 3), halo=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin={"field1": (2, 2, 2)}, domain=(1, 1, 1))

    assert A[2, 2, 2] == 4
    assert B[2, 2, 2] == 7
    assert C[0, 1, 0] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47


def test_domain_selection():
    A = gt_storage.ones(defaults="gtmc", dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    B = gt_storage.ones(defaults="gtx86", dtype=np.float64, shape=(3, 3, 3), halo=(2, 2, 2))
    C = gt_storage.ones(defaults="numpy", dtype=np.float32, shape=(3, 3, 3), halo=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    assert A[1, 1, 1] == 4
    assert B[1, 1, 1] == 7
    assert C[1, 1, 1] == 21
    assert np.sum(np.asarray(A)) == 30
    assert np.sum(np.asarray(B)) == 33
    assert np.sum(np.asarray(C)) == 47

    A = gt_storage.ones(defaults="gtmc", dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    B = gt_storage.ones(defaults="gtx86", dtype=np.float64, shape=(3, 3, 3), halo=(2, 2, 2))
    C = gt_storage.ones(defaults="numpy", dtype=np.float32, shape=(3, 3, 3), halo=(0, 1, 0))
    stencil(A, B, C, param=3.0, origin=(0, 0, 0))

    assert np.all(np.asarray(A) == 4)
    assert np.all(np.asarray(B) == 7)
    assert np.all(np.asarray(C) == 21)


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

    arg1 = gt_storage.ones(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    arg2 = gt_storage.zeros(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    arg3 = gt_storage.ones(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, None, arg3, par1=2.0)
    np.testing.assert_equal(arg1, 14 * np.ones((3, 3, 3)))
    branch_true(arg1, None, par1=2.0)
    np.testing.assert_equal(arg1, 196 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par3=2.0)
    np.testing.assert_equal(arg1, 56 * np.ones((3, 3, 3)))
    try:
        branch_false(arg1, arg2, par1=2.0, par3=2.0)
    except ValueError:
        pass
    else:
        assert False

    arg1 = gt_storage.ones(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    arg2 = gt_storage.zeros(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    arg3 = gt_storage.ones(defaults=backend, dtype=np.float64, shape=(3, 3, 3), halo=(0, 0, 0))
    tmp = np.asarray(arg3)
    tmp *= 2

    branch_true(arg1, arg2=None, par1=2.0, par2=5.0, par3=3.0)
    np.testing.assert_equal(arg1, 10 * np.ones((3, 3, 3)))
    branch_true(arg1, arg2=None, par1=2.0, par2=5.0)
    np.testing.assert_equal(arg1, 100 * np.ones((3, 3, 3)))
    branch_false(arg1, arg2, arg3, par1=2.0, par2=5.0, par3=3.0)
    np.testing.assert_equal(arg1, 60 * np.ones((3, 3, 3)))
    try:
        branch_false(arg1, arg2, arg3, par1=2.0, par2=5.0)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.parametrize("backend", INTERNAL_CPU_BACKENDS)
def test_halo_checks(backend):
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test default works
    in_field = gt_storage.ones(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field)
    assert (np.asarray(out_field[1:-1, 1:-1, :]) == 1).all()

    # test setting arbitrary, small domain works
    in_field = gt_storage.ones(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(10, 10, 10))
    assert (np.asarray(out_field[2:12, 2:12, :]) == 1).all()

    # test setting domain+origin too large raises
    in_field = gt_storage.ones(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        defaults=backend, shape=(22, 22, 10), halo=(1, 1, 0), dtype=np.float64
    )
    with pytest.raises(ValueError):
        stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))

    # test 2*origin+domain does not raise if still fits (c.f. previous bug in c++ check.)
    in_field = gt_storage.ones(
        defaults=backend, shape=(23, 23, 10), halo=(1, 1, 0), dtype=np.float64
    )
    out_field = gt_storage.zeros(
        defaults=backend, shape=(23, 23, 10), halo=(1, 1, 0), dtype=np.float64
    )
    stencil(in_field=in_field, out_field=out_field, origin=(2, 2, 0), domain=(20, 20, 10))


def test_np_int_types():
    backend = "numpy"
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    # test numpy int types are accepted
    in_field = gt_storage.ones(
        defaults=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        halo=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        defaults=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        halo=(np.int64(1), int(1), 0),
        dtype=np.float64,
    )
    stencil(
        in_field=in_field,
        out_field=out_field,
        origin=(np.int8(2), np.int16(2), np.int32(0)),
        domain=(np.int64(20), int(20), 10),
    )


@pytest.mark.parametrize("backend", ["debug", "numpy", "gtx86"])
def test_exec_info(backend):
    """test that proper warnings are raised depending on field type."""
    stencil = gtscript.stencil(definition=avg_stencil, backend=backend)

    exec_info = {}
    # test numpy int types are accepted
    in_field = gt_storage.ones(
        defaults=backend,
        shape=(np.int8(23), np.int16(23), np.int32(10)),
        halo=(1, 1, 0),
        dtype=np.float64,
    )
    out_field = gt_storage.zeros(
        defaults=backend, shape=(23, 23, 10), halo=(1, 1, 0), dtype=np.float64
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


@pytest.mark.parametrize(
    "backend",
    INTERNAL_CPU_BACKENDS
    + [pytest.param(b, marks=[pytest.mark.requires_gpu]) for b in INTERNAL_GPU_BACKENDS],
)
class TestNonStorageArguments:

    stencils = {}

    def stencil(self, backend):
        if backend in self.stencils:
            return self.stencils[backend]

        @gtscript.stencil(backend=backend)
        def stencil(field: gtscript.Field[np.float64]):
            with computation(PARALLEL), interval(...):
                field = 3.0

        self.stencils[backend] = stencil
        return stencil

    def test_array_interface(self, backend):
        storage = gt_storage.ones((3, 3, 3), defaults=backend, device="cpu", managed=False)
        assert isinstance(storage, gt_storage.definitions.CPUStorage)

        stencil = self.stencil(backend)
        stencil(storage._field)

        np.testing.assert_equal(storage.to_numpy(), 3.0)

    @pytest.mark.requires_gpu
    def test_cuda_array_interface(self, backend):
        storage = gt_storage.ones((3, 3, 3), defaults=backend, device="gpu", managed=False)
        assert isinstance(storage, gt_storage.definitions.GPUStorage)

        stencil = self.stencil(backend)
        stencil(storage._device_field)

        np.testing.assert_equal(storage.to_numpy(), 3.0)

    @pytest.mark.parametrize(
        "field_class",
        [
            "custom",
            pytest.param(
                "xarray",
                marks=[
                    pytest.mark.skip(reason="Not implemented"),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize(
        ["device", "managed"],
        [
            ("cpu", False),
            pytest.param("gpu", False, marks=[pytest.mark.requires_gpu]),
            pytest.param("gpu", "cuda", marks=[pytest.mark.requires_gpu]),
            pytest.param("gpu", "gt4py", marks=[pytest.mark.requires_gpu]),
        ],
    )
    def test_gt_data_interface(self, backend, field_class, device, managed):
        storage = gt_storage.ones((3, 3, 3), defaults=backend)

        if field_class == "xarray":
            raise NotImplementedError
        elif field_class == "custom":

            class Wrapper:
                def __init__(self, storage):
                    self._storage = storage

                @property
                def __gt_data_interface__(self):
                    return self._storage.__gt_data_interface__

            field = Wrapper(storage)
        else:
            raise ValueError("Bad test parametrization.")

        stencil = self.stencil(backend)
        stencil(field)

        np.testing.assert_equal(storage.to_numpy(), 3.0)
