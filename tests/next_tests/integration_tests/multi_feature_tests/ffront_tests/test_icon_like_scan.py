# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common

from next_tests import definitions as test_definitions
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import Cell, KDim, Koff
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


pytestmark = [pytest.mark.uses_unstructured_shift, pytest.mark.uses_scan]


Cell = gtx.Dimension("Cell")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
Koff = gtx.FieldOffset("Koff", KDim, (KDim,))


@gtx.scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, True))
def _scan(
    state: tuple[float, float, bool], w: float, z_q: float, z_a: float, z_b: float, z_c: float
) -> tuple[float, float, bool]:
    z_q_m1, w_m1, first = state
    z_g = z_b + z_a * z_q_m1
    z_q_new = (0.0 - z_c) * z_g
    w_new = z_a * w_m1 * z_g
    return (z_q, w, False) if first else (z_q_new, w_new, False)


@gtx.field_operator
def _solve_nonhydro_stencil_52_like(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
) -> tuple[
    gtx.Field[[Cell, KDim], float], gtx.Field[[Cell, KDim], float], gtx.Field[[Cell, KDim], bool]
]:
    """No projector required as we write all output of the scan (including dummy field)"""
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, dummy = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res, w_res, dummy


@gtx.program
def solve_nonhydro_stencil_52_like(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
    dummy: gtx.Field[[Cell, KDim], bool],
):
    _solve_nonhydro_stencil_52_like(
        z_alpha, z_beta, z_q, w, out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:])
    )


@gtx.field_operator
def _solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
) -> tuple[gtx.Field[[Cell, KDim], float], gtx.Field[[Cell, KDim], float]]:
    """In inlining, relies on CollapseTuple with ignore_tuple_size=True (only working with gtfn)."""
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res, w_res


@gtx.program
def solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(
        z_alpha, z_beta, z_q, w, out=(z_q[:, 1:], w[:, 1:])
    )


@gtx.field_operator
def _solve_nonhydro_stencil_52_like_z_q(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
) -> gtx.Field[[Cell, KDim], float]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res


@gtx.program
def solve_nonhydro_stencil_52_like_z_q(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
    z_q_out: gtx.Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_z_q(z_alpha, z_beta, z_q, w, out=z_q_out[:, 1:])


@gtx.field_operator
def _solve_nonhydro_stencil_52_like_z_q_tup(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
) -> tuple[gtx.Field[[Cell, KDim], float]]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return (z_q_res,)


@gtx.program
def solve_nonhydro_stencil_52_like_z_q_tup(
    z_alpha: gtx.Field[[Cell, KDim], float],
    z_beta: gtx.Field[[Cell, KDim], float],
    z_q: gtx.Field[[Cell, KDim], float],
    w: gtx.Field[[Cell, KDim], float],
    z_q_out: gtx.Field[[Cell, KDim], float],
):
    _solve_nonhydro_stencil_52_like_z_q_tup(z_alpha, z_beta, z_q, w, out=(z_q_out[:, 1:],))


def reference(
    z_alpha: np.array, z_beta: np.array, z_q_ref: np.array, w_ref: np.array
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
def test_setup(exec_alloc_descriptor):
    test_case = cases.Case(
        None
        if isinstance(exec_alloc_descriptor, test_definitions.EmbeddedDummyBackend)
        else exec_alloc_descriptor,
        offset_provider={"Koff": KDim},
        default_sizes={Cell: 14, KDim: 10},
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )

    @dataclasses.dataclass(frozen=True)
    class setup:
        case: cases.Case = dataclasses.field(default_factory=lambda: test_case)
        cell_size = test_case.default_sizes[Cell]
        k_size = test_case.default_sizes[KDim]
        z_alpha = test_case.as_field(
            [Cell, KDim], np.random.default_rng().uniform(size=(cell_size, k_size + 1))
        )
        z_beta = test_case.as_field(
            [Cell, KDim], np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        z_q = test_case.as_field(
            [Cell, KDim], np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        w = test_case.as_field(
            [Cell, KDim], np.random.default_rng().uniform(size=(cell_size, k_size))
        )
        z_q_ref, w_ref = reference(z_alpha.ndarray, z_beta.ndarray, z_q.ndarray, w.ndarray)
        dummy = test_case.as_field([Cell, KDim], np.zeros((cell_size, k_size), dtype=bool))
        z_q_out = test_case.as_field([Cell, KDim], np.zeros((cell_size, k_size)))

    return setup()


@pytest.mark.uses_tuple_returns
@pytest.mark.uses_scan_requiring_projector
def test_solve_nonhydro_stencil_52_like_z_q(test_setup):
    cases.verify(
        test_setup.case,
        solve_nonhydro_stencil_52_like_z_q,
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.z_q_out,
        ref=test_setup.z_q_ref,
        inout=test_setup.z_q_out,
        comparison=lambda ref, a: np.allclose(ref[:, 1:], a[:, 1:]),
    )

    assert np.allclose(test_setup.z_q_ref[:, 1:], test_setup.z_q_out[:, 1:].asnumpy())


@pytest.mark.uses_tuple_returns
def test_solve_nonhydro_stencil_52_like_z_q_tup(test_setup):
    if test_setup.case.backend == test_definitions.ProgramBackendId.ROUNDTRIP.load():
        pytest.xfail("Needs proper handling of tuple[Column] <-> Column[tuple].")

    cases.verify(
        test_setup.case,
        solve_nonhydro_stencil_52_like_z_q_tup,
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.z_q_out,
        ref=test_setup.z_q_ref,
        inout=test_setup.z_q_out,
        comparison=lambda ref, a: np.allclose(ref[:, 1:], a[:, 1:]),
    )


@pytest.mark.uses_tuple_returns
def test_solve_nonhydro_stencil_52_like(test_setup):
    cases.run(
        test_setup.case,
        solve_nonhydro_stencil_52_like,
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        test_setup.dummy,
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q.asnumpy())
    assert np.allclose(test_setup.w_ref, test_setup.w.asnumpy())


@pytest.mark.uses_tuple_returns
def test_solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge(test_setup):
    if test_setup.case.backend == test_definitions.ProgramBackendId.ROUNDTRIP.load():
        pytest.xfail("Needs proper handling of tuple[Column] <-> Column[tuple].")

    cases.run(
        test_setup.case,
        solve_nonhydro_stencil_52_like_with_gtfn_tuple_merge,
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q.asnumpy())
    assert np.allclose(test_setup.w_ref, test_setup.w.asnumpy())
