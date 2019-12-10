# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import gt4py as gt
import gt4py.gtscript as gtscript
import gt4py.backend as gt_backend
import gt4py.storage as gt_storage
from gt4py.gtscript import Field
import pytest


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


def avg_stencil(in_field: Field[np.float64], out_field: Field[np.float64]):
    with computation(PARALLEL), interval(...):
        out_field = 0.25 * (
            +in_field[0, 1, 0] + in_field[0, -1, 0] + in_field[1, 0, 0] + in_field[-1, 0, 0]
        )


@pytest.mark.parametrize(
    "backend",
    [
        name
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "cpu"
    ],
)
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
    try:
        branch_false(arg1, arg2, par1=2.0, par3=2.0)
    except ValueError:
        pass
    else:
        assert False

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
    try:
        branch_false(arg1, arg2, arg3, par1=2.0, par2=5.0)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.parametrize(
    "backend",
    [
        name
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] != "gpu"
    ],
)
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
