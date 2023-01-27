from dataclasses import dataclass

import numpy as np
import pytest

from functional.common import Dimension, DimensionKind, Field
from functional.ffront.decorator import field_operator, program, scan_operator
from functional.ffront.fbuiltins import FieldOffset
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners import gtfn_cpu, roundtrip


# TODO duplicated from test_execution
@pytest.fixture(params=[roundtrip.executor, gtfn_cpu.run_gtfn])
def fieldview_backend(request):
    yield request.param


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
    z_q_new = -z_c * z_g
    w_new = z_a * w_m1 * z_g
    return (z_q, w, False) if first else (z_q_new, w_new, False)


@field_operator
def _solve_nonhydro_stencil_52_like(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> tuple[Field[[Cell, KDim], float], Field[[Cell, KDim], float]]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return z_q_res, w_res


@program
def solve_nonhydro_stencil_52_like(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
):

    _solve_nonhydro_stencil_52_like(
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
):

    _solve_nonhydro_stencil_52_like_z_q(z_alpha, z_beta, z_q, w, out=z_q[:, 1:])


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
):

    _solve_nonhydro_stencil_52_like_z_q_tup(z_alpha, z_beta, z_q, w, out=(z_q[:, 1:],))


@field_operator
def _solve_nonhydro_stencil_52_like_w(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
) -> Field[[Cell, KDim], float]:
    z_a = z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = z_beta * z_alpha(Koff[1])
    z_b = z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q_res, w_res, _ = _scan(w, z_q, z_a, z_b, z_c)
    return w_res


@program
def solve_nonhydro_stencil_52_like_w(
    z_alpha: Field[[Cell, KDim], float],
    z_beta: Field[[Cell, KDim], float],
    z_q: Field[[Cell, KDim], float],
    w: Field[[Cell, KDim], float],
):

    _solve_nonhydro_stencil_52_like_w(
        z_alpha,
        z_beta,
        z_q,
        w,
        out=w[:, 1:],
    )


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

    return setup()


def test_solve_nonhydro_stencil_52_like_z_q(test_setup, fieldview_backend):
    solve_nonhydro_stencil_52_like_z_q.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q)


def test_solve_nonhydro_stencil_52_like_z_q_tup(test_setup, fieldview_backend):
    solve_nonhydro_stencil_52_like_z_q_tup.with_backend(fieldview_backend)(
        test_setup.z_alpha,
        test_setup.z_beta,
        test_setup.z_q,
        test_setup.w,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(test_setup.z_q_ref, test_setup.z_q)


# def test_solve_nonhydro_stencil_52_like_w(test_setup):
#     solve_nonhydro_stencil_52_like_w(
#         test_setup.z_alpha,
#         test_setup.z_beta,
#         test_setup.z_q,
#         test_setup.w,
#         offset_provider={"Koff": KDim},
#     )

#     assert np.allclose(test_setup.w_ref, test_setup.w)


# def test_solve_nonhydro_stencil_52_like(test_setup):
#     solve_nonhydro_stencil_52_like(
#         test_setup.z_alpha,
#         test_setup.z_beta,
#         test_setup.z_q,
#         test_setup.w,
#         offset_provider={"Koff": KDim},
#     )

#     assert np.allclose(test_setup.z_q_ref, test_setup.z_q)
#     assert np.allclose(test_setup.w_ref, test_setup.w)
