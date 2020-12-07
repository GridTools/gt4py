# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import Field

from ..definitions import ALL_BACKENDS


DAWN_BACKENDS = tuple(filter(lambda name: "dawn:" in name, ALL_BACKENDS))


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


@pytest.mark.parametrize("backend", DAWN_BACKENDS)
def test_assert_same_shape(backend):
    stencil_call = gtscript.stencil(definition=stencil, backend=backend)

    A = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(0, 0, 0)
    )
    B = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(3, 3, 3), default_origin=(2, 2, 2)
    )
    C = gt_storage.ones(
        backend=backend, dtype=np.float32, shape=(3, 3, 3), default_origin=(0, 1, 0)
    )
    stencil_call(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    stencil_call(
        A,
        B,
        C,
        param=3.0,
        origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
        domain=(1, 1, 1),
    )

    A = gt_storage.ones(
        backend=backend, dtype=np.float64, shape=(5, 5, 5), default_origin=(0, 0, 0)
    )
    A = A[1:-1, 1:-1, 1:-1]
    A.is_stencil_view = True
    stencil_call(
        A,
        B,
        C,
        param=3.0,
        origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
        domain=(1, 1, 1),
    )

    C = gt_storage.ones(
        backend=backend, dtype=np.float32, shape=(5, 5, 5), default_origin=(0, 1, 0)
    )
    with pytest.raises(ValueError):
        stencil_call(A, B, C, param=3.0, origin=(1, 1, 1), domain=(1, 1, 1))

    with pytest.raises(ValueError):
        stencil_call(
            A,
            B,
            C,
            param=3.0,
            origin=dict(field1=(2, 2, 2), field2=(0, 0, 0), field3=(1, 1, 1)),
            domain=(1, 1, 1),
        )
