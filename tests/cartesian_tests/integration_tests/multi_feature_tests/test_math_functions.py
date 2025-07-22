# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import pytest

from gt4py import storage
from gt4py.storage.cartesian import utils as storage_utils
from gt4py.cartesian.gtscript import (
    computation,
    PARALLEL,
    erf,
    erfc,
    interval,
    round,
    stencil,
    Field,
)

from ...definitions import ALL_BACKENDS, get_array_library


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_erf(backend):
    xp = get_array_library(backend)  # numpy or cupy depending on backend

    @stencil(backend=backend)
    def stencil_erf(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = erf(field_a)

    initial_values = xp.array([-1, 0, 1, 2], dtype=xp.float32)
    array_shape = (1, 1, len(initial_values))
    A = storage.zeros(array_shape, np.float32, backend=backend)
    A[:, :, :] = initial_values
    B = storage.full(array_shape, 42.0, np.float32, backend=backend)

    stencil_erf(A, B)

    expected = np.array([math.erf(n) for n in initial_values], dtype=xp.float32)
    assert (A[0, 0, :] == initial_values).all()
    B_ = storage_utils.cpu_copy(B)
    np.testing.assert_allclose(B_[0, 0, :], expected)  # gpu generates slightly different values


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_erfc(backend):
    xp = get_array_library(backend)  # numpy or cupy depending on backend

    @stencil(backend=backend)
    def stencil_erfc(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = erfc(field_a)

    initial_values = xp.array([-1, 0, 1, 2], dtype=xp.float32)
    array_shape = (1, 1, len(initial_values))
    A = storage.zeros(array_shape, np.float32, backend=backend)
    A[:, :, :] = initial_values
    B = storage.full(array_shape, 42.0, np.float32, backend=backend)

    stencil_erfc(A, B)

    expected = np.array([math.erfc(n) for n in initial_values], dtype=xp.float32)
    assert (A[0, 0, :] == initial_values).all()
    B_ = storage_utils.cpu_copy(B)
    np.testing.assert_allclose(B_[0, 0, :], expected)  # gpu generates slightly different values


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_round(backend):
    xp = get_array_library(backend)  # numpy or cupy depending on backend

    @stencil(backend=backend)
    def stencil_round(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = round(field_a)

    initial_values = xp.array([-1.5, -0.5, 0.3, 0.5, 0.8, 1.2, 1.5], dtype=xp.float32)
    array_shape = (1, 1, len(initial_values))
    A = storage.zeros(array_shape, np.float32, backend=backend)
    A[:, :, :] = initial_values
    B = storage.full(array_shape, -1.0, np.float32, backend=backend)

    stencil_round(A, B)

    expected = xp.array([-2.0, -1.0, 0.0, 1.0, 1.0, 1.0, 2.0], dtype=xp.float32)
    assert (A[0, 0, :] == initial_values).all()
    assert (B[0, 0, :] == expected).all()
