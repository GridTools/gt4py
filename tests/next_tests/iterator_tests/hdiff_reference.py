# GT4Py - GridTools Framework
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
import pytest


def hdiff_reference_impl():
    shape = (5, 7, 5)
    rng = np.random.default_rng()
    inp = rng.normal(size=(shape[0] + 4, shape[1] + 4, shape[2]))
    coeff = rng.normal(size=shape)

    lap = 4 * inp[1:-1, 1:-1, :] - (
        inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] + inp[1:-1, 2:, :] + inp[1:-1, :-2, :]
    )
    uflx = lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]
    flx = np.where(uflx * (inp[2:-1, 2:-2, :] - inp[1:-2, 2:-2, :]) > 0, 0, uflx)
    ufly = lap[1:-1, 1:, :] - lap[1:-1, :-1, :]
    fly = np.where(ufly * (inp[2:-2, 2:-1, :] - inp[2:-2, 1:-2, :]) > 0, 0, ufly)
    out = inp[2:-2, 2:-2, :] - coeff * (
        flx[1:, :, :] - flx[:-1, :, :] + fly[:, 1:, :] - fly[:, :-1, :]
    )

    return inp, coeff, out


@pytest.fixture
def hdiff_reference():
    return hdiff_reference_impl()
