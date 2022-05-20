# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


def test_masked_vector_assignment():
    from gt4py.gtscript import FORWARD, IJ, Field, computation, interval, stencil
    from gt4py.storage import ones

    BACKEND = "numpy"
    dtype = np.float64

    @stencil(BACKEND)
    def masked_vector_assignment(fld2D: Field[IJ, dtype]):

        with computation(FORWARD), interval(0, None):
            fld2D += fld2D
            if fld2D >= 1.0:
                fld2D = 0.0

    origin = (0, 0, 0)
    fld2D = ones(shape=(2, 3), dtype=dtype, backend=BACKEND, default_origin=origin)

    masked_vector_assignment(fld2D)

    assert np.allclose(fld2D, np.zeros((2, 3)))
