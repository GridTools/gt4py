# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

import numpy as np


def test_masked_vector_assignment():
    from gt4py.cartesian.gtscript import FORWARD, IJ, Field, computation, interval, stencil
    from gt4py.storage import ones

    BACKEND = "numpy"
    dtype = np.float64

    @stencil(BACKEND)
    @typing.no_type_check
    def masked_vector_assignment(fld2D: Field[IJ, dtype]):
        with computation(FORWARD), interval(0, None):
            fld2D += fld2D
            if fld2D >= 1.0:
                fld2D = 0.0

    origin = (0, 0)
    fld2D = ones(shape=(2, 3), dtype=dtype, backend=BACKEND, aligned_index=origin)

    masked_vector_assignment(fld2D)

    assert np.allclose(fld2D, np.zeros((2, 3)))
