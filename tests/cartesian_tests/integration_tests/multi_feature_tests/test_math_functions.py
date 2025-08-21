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
from gt4py.cartesian.gtscript import (
    computation,
    PARALLEL,
    erf,
    erfc,
    interval,
    round,
    round_away_from_zero,
    stencil,
    Field,
)

from ...definitions import ALL_BACKENDS


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_erf(backend):
    @stencil(backend=backend)
    def stencil_erf(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = erf(field_a)

    initial_values = np.array([[[-1, 0, 1, 2]]], dtype=np.float32)
    A = storage.from_array(initial_values, np.float32, backend=backend)
    B = storage.full(initial_values.shape, 42.0, np.float32, backend=backend)

    stencil_erf(A, B)

    expected = np.array([math.erf(n) for n in initial_values[0, 0, :]], dtype=np.float32)

    assert (A[0, 0, :] == initial_values).all()
    np.testing.assert_allclose(B[0, 0, :], expected)  # gpu generates slightly different values


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_erfc(backend):
    @stencil(backend=backend)
    def stencil_erfc(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = erfc(field_a)

    initial_values = np.array([[[-1, 0, 1, 2]]], dtype=np.float32)
    A = storage.from_array(initial_values, np.float32, backend=backend)
    B = storage.full(initial_values.shape, 42.0, np.float32, backend=backend)

    stencil_erfc(A, B)

    expected = np.array([math.erfc(n) for n in initial_values[0, 0, :]], dtype=np.float32)
    assert (A[0, 0, :] == initial_values).all()
    np.testing.assert_allclose(B[0, 0, :], expected)  # gpu generates slightly different values


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_round(backend):
    @stencil(backend=backend)
    def stencil_round(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = round(field_a)

    initial_values = np.array([[[-1.5, -0.5, 0.3, 0.5, 0.8, 1.2, 1.5]]], dtype=np.float32)
    A = storage.from_array(initial_values, np.float32, backend=backend)
    B = storage.full(initial_values.shape, -1.0, np.float32, backend=backend)

    stencil_round(A, B)

    expected = np.array([-2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    assert (A[0, 0, :] == initial_values).all()
    assert (B[0, 0, :] == expected).all()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_round_away_from_zero(backend):
    @stencil(backend=backend)
    def stencil_round(field_a: Field[np.float32], field_b: Field[np.float32]):
        with computation(PARALLEL), interval(...):
            field_b = round_away_from_zero(field_a)

    initial_values = np.array([[[-1.5, -0.5, 0.3, 0.5, 0.8, 1.2, 1.5]]], dtype=np.float32)
    A = storage.from_array(initial_values, np.float32, backend=backend)
    B = storage.full(initial_values.shape, -1.0, np.float32, backend=backend)

    stencil_round(A, B)

    expected = np.array([-2.0, -1.0, 0.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    assert (A[0, 0, :] == initial_values).all()
    assert (B[0, 0, :] == expected).all()
