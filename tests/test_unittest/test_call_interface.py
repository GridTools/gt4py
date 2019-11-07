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

import gt4py.gtscript as gtscript
from gt4py.gtscript import Field
import gt4py.backend as gt_backend
import gt4py.storage as gt_storage


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


def test_assert_same_shape():
    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
    stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    B = gt_storage.ones(
        backend="gtx86", dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend="numpy", dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
    stencil(
        A,
        B,
        C,
        param=3.0,
        origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
        domain=(1, 1, 1),
    )
    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(5, 5, 5), default_origin=(0, 0, 0)
    )
    A = A[1:-1, 1:-1, 1:-1]
    A.is_stencil_view = True
    stencil(
        A,
        B,
        C,
        param=3.0,
        origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
        domain=(1, 1, 1),
    )

    try:
        C = gt_storage.ones(
            backend="numpy", dtype=np.float32, shape=(5, 5, 5), default_origin=(0, 1, 0)
        )
        stencil(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))
    except ValueError:
        pass
    else:
        assert False

    try:

        C = gt_storage.ones(
            backend="numpy", dtype=np.float32, shape=(5, 5, 5), default_origin=(0, 1, 0)
        )
        stencil(
            A,
            B,
            C,
            param=3.0,
            origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
            domain=(1, 1, 1),
        )
    except ValueError:
        pass
    else:
        assert False


def test_origin_selection():
    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
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

    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
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

    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
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


def test_domain_selection():
    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
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

    A = gt_storage.ones(
        backend="gtmc", dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
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
