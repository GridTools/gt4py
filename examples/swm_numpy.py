# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
NumPy reference for the shallow water model example.

Companion to `swm.ipynb`. The model is doubly periodic, so periodicity is expressed here
with `numpy.roll` on interior-only arrays of shape `(M, N)` -- there is no halo. The
notebook's GT4Py version instead carries one halo cell per side and updates it explicitly,
which is what makes the two implementations worth comparing.
"""

from __future__ import annotations

import numpy as np


# Model constants of the NCAR/SWM benchmark configuration.
M = 16
N = 16
DX = 100000.0
DY = 100000.0
DT = 90.0
ALPHA = 0.001
A = 1000000.0
P_MEAN = 50000.0
ITMAX = 4000


def avg_x(f: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(f, -1, axis=0) + f)


def avg_y(f: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(f, -1, axis=1) + f)


def avg_x_staggered(f: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(f, 1, axis=0) + f)


def avg_y_staggered(f: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(f, 1, axis=1) + f)


def delta_x(dx: float, f: np.ndarray) -> np.ndarray:
    return (np.roll(f, -1, axis=0) - f) / dx


def delta_y(dy: float, f: np.ndarray) -> np.ndarray:
    return (np.roll(f, -1, axis=1) - f) / dy


def delta_x_staggered(dx: float, f: np.ndarray) -> np.ndarray:
    return (f - np.roll(f, 1, axis=0)) / dx


def delta_y_staggered(dy: float, f: np.ndarray) -> np.ndarray:
    return (f - np.roll(f, 1, axis=1)) / dy


def initial_conditions(
    m: int = M, n: int = N, dx: float = DX, dy: float = DY, a: float = A, p_mean: float = P_MEAN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the initial `u`, `v`, `p` on the interior grid, each of shape `(m, n)`.

    The velocity field is derived from a doubly periodic stream function
    `psi = a * sin(2*pi*(i+0.5)/m) * sin(2*pi*(j+0.5)/n)`.
    """
    d_i = 2.0 * np.pi / m
    d_j = 2.0 * np.pi / n
    el = n * dx
    pcf = (np.pi * np.pi * a * a) / (el * el)

    psi = (
        a
        * np.sin((np.arange(0, m + 1).reshape(-1, 1) + 0.5) * d_i)
        * np.sin((np.arange(0, n + 1) + 0.5) * d_j)
    )
    p = (
        pcf
        * (np.cos(2.0 * np.arange(0, m).reshape(-1, 1) * d_i) + np.cos(2.0 * np.arange(0, n) * d_j))
        + p_mean
    )
    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def timestep(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    uold: np.ndarray,
    vold: np.ndarray,
    pold: np.ndarray,
    dx: float,
    dy: float,
    tdt: float,
    alpha: float,
) -> tuple[np.ndarray, ...]:
    """One leapfrog step with a Robert-Asselin time filter. Returns the six updated fields."""
    cu = avg_x(p) * u
    cv = avg_y(p) * v
    z = (delta_x(dx, v) - delta_y(dy, u)) / avg_x(avg_y(p))
    h = p + 0.5 * (avg_x_staggered(u * u) + avg_y_staggered(v * v))

    unew = uold + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * tdt - delta_x(dx, h) * tdt
    vnew = vold - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * tdt - delta_y(dy, h) * tdt
    pnew = pold - delta_x_staggered(dx, cu) * tdt - delta_y_staggered(dy, cv) * tdt

    uold_new = u + alpha * (unew - 2.0 * u + uold)
    vold_new = v + alpha * (vnew - 2.0 * v + vold)
    pold_new = p + alpha * (pnew - 2.0 * p + pold)

    return unew, vnew, pnew, uold_new, vold_new, pold_new


def run(
    itmax: int,
    m: int = M,
    n: int = N,
    dx: float = DX,
    dy: float = DY,
    dt: float = DT,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate `itmax` steps and return the final interior `u`, `v`, `p`."""
    u, v, p = initial_conditions(m, n, dx, dy)
    uold, vold, pold = u.copy(), v.copy(), p.copy()

    for cycle in range(itmax):
        # The first step is forward Euler and unfiltered; afterwards leapfrog with tdt = 2*dt.
        tdt = dt if cycle == 0 else 2.0 * dt
        step_alpha = 0.0 if cycle == 0 else alpha
        unew, vnew, pnew, uold, vold, pold = timestep(
            u, v, p, uold, vold, pold, dx, dy, tdt, step_alpha
        )
        u, v, p = unew, vnew, pnew

    return u, v, p


def to_reference_layout(
    u: np.ndarray, v: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert interior `(M, N)` fields to the `(M+1, N+1)` layout of the NCAR/SWM dumps.

    The extra row and column are the periodic images the original Fortran carried
    explicitly; which side they sit on differs per field because of the C-grid staggering.
    """
    return (
        np.pad(u, ((1, 0), (0, 1)), mode="wrap"),
        np.pad(v, ((0, 1), (1, 0)), mode="wrap"),
        np.pad(p, ((0, 1), (0, 1)), mode="wrap"),
    )
