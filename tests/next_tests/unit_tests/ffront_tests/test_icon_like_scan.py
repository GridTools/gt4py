# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from dataclasses import dataclass

import numpy as np
import pytest

from gt4py.next.common import Dimension, DimensionKind, Field
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners import gtfn_cpu, roundtrip


Cell = Dimension("Cell")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)
Koff = FieldOffset("Koff", KDim, (KDim,))


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, True))
def _scan(
    state: tuple[float, float, bool],
    w: float,
    z_q: float,
    z_a: float,
    z_b: float,
    z_c: float,
) -> tuple[float, float, bool]:
    z_q_m1, w_m1, first = state
    z_g = z_b + z_a * z_q_m1
    z_q_new = (0.0 - z_c) * z_g
    w_new = z_a * w_m1 * z_g
    return (z_q, w, False) if first else (z_q_new, w_new, False)


@field_operator
def _solve_nonhydro_stencil_52_like(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> tuple[Field[[Cell, KDim], float], Field[[Cell, KDim], float], Field[[Cell, KDim], bool]]:
    """No projector required as we write all output of the scan (including dummy field)"""
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, dummy = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res, w_res, dummy


@program
def solve_nonhydro_stencil_52_like(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
    dummy: Field[[Cell, KDim], bool],
):
    _solve_nonhydro_stencil_52_like(
        z_alpha,
        z_beta,
        z_q,
        w,
        out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:]),
    )


@field_operator
def _solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> tuple[Field[[Cell, KDim], float], Field[[Cell, KDim], float]]:
    """In inlining, relies on CollapseTuple with ignore_tuple_size=True (only working with gtfn)."""
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res, w_res


@program
def solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
        z_alpha,
        z_beta,
        z_q,
        w,
        out=(z_q[:, 1:], w[:, 1:]),
    )


@field_operator
def _solve_nonhydro_stencil_52_like_z_q(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> Field[[Cell, KDim], float]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res


@program
def solve_nonhydro_stencil_52_like_z_q(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
    z_q_out: Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_z_q(z_alpha, z_beta, z_q, w, out=z_q_out[:, 1:])


@field_operator
def _solve_nonhydro_stencil_52_like_z_q_tup(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> tuple[Field[[Cell, KDim], float]]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return (z_q_res,)


@program
def solve_nonhydro_stencil_52_like_z_q_tup(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
    z_q_out: Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_z_q_tup(z_alpha, z_beta, z_q, w, out=(z_q_out[:, 1:],))


def reference(
    z_alpha: np.array,
    z_beta: np.array,
    z_q_ref: np.array,
    w_ref: np.array,
) -> tuple[np.ndarray, np.ndarray]:
    z_q = np.copy(z_q_ref)
    w = np.copy(w_ref)
    z_a = np.zeros_like(z_beta)
    z_b = np.zeros_like(z_beta)
    z_c = np.zeros_like(z_beta)
    z_g = np.zeros_like(z_beta)

    k_size = w.shape[1]
    for k in range(2, k_size):
        z_a[:, k] = z_beta[:, k - 1] * z_alpha[:, k - 1]
        z_c[:, k] = z_beta[:, k] * z_alpha[:, k + 1]
        z_b[:, k] = z_alpha[:, k] * (z_beta[:, k - 1] + z_beta[:, k])
        z_g[:, k] = z_b[:, k] + z_a[:, k] * z_q[:, k - 1]
        z_q[:, k] = -z_c[:, k] * z_g[:, k]

        w[:, k] = (z_a[:, k] * w[:, k - 1]) * z_g[:, k]
    return z_q, w


@pytest.fixture
def test_setup():
    @dataclass(frozen=True)
    class setup:
        cell_size = 14
        k_size = 10
        z_alpha = np_as_located_field(Cell, KDim)(
            np.random.default_rng().uniform(size=(cell_size, k_size + 1))
        )
        z_beta = np_as_located_field(Cell, KDim)(
            np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        z_q = np_as_located_field(Cell, KDim)(
            np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        w = np_as_located_field(Cell, KDim)(
            np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        z_q_ref, w_ref = reference(z_alpha, z_beta, z_q, w)
        dummy = np_as_located_field(Cell, KDim)(np.zeros((cell_size, k_size), dtype=bool))
        z_q_out = np_as_located_field(Cell, KDim)(np.zeros((cell_size, k_size)))

    return setup()


def test_solve_nonhydro_stencil_52_like_z_q(test_setup, fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip("Needs implementation of scan projector.")
    if fieldview_backend == roundtrip.executor:
        pytest.skip("Inline into scan breaks embedded execution.")

    solve_nonhydro_stencil_52_like_z_q.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.z_q_out,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref[:, 1:], test_setup.z_q_out[:, 1:])


def test_solve_nonhydro_stencil_52_like_z_q_tup(test_setup, fieldview_backend):
    if fieldview_backend == roundtrip.executor:
        pytest.skip(
            "Inline into scan breaks embedded execution and relies on CollapseTuple ignore_tuple_size==True."
        )

    solve_nonhydro_stencil_52_like_z_q_tup.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.z_q_out,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref[:, 1:], test_setup.z_q_out[:, 1:])


def test_solve_nonhydro_stencil_52_like(test_setup, fieldview_backend):
    if fieldview_backend == roundtrip.executor:
        pytest.skip("Inline into scan breaks embedded execution.")

    solve_nonhydro_stencil_52_like.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.dummy,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q)
    assert np.allclose(test_setup.w_ref, test_setup.w)


def test_solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(test_setup, fieldview_backend):
    if fieldview_backend == roundtrip.executor:
        pytest.skip("Only working in gtfn with CollapseTuple ignore_tuple_size==True.")

    solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q)
    assert np.allclose(test_setup.w_ref, test_setup.w)
