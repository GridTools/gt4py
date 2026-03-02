# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def initialize_interior(xp, M, N, dx, dy, a):
    pi = 4.0 * xp.arctan(1.0)
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    psi = (
        a
        * xp.sin((xp.arange(0, M + 1).reshape(-1, 1) + 0.5) * d_i)
        * xp.sin((xp.arange(0, N + 1) + 0.5) * d_j)
    )
    p = (
        pcf
        * (xp.cos(2.0 * xp.arange(0, M).reshape(-1, 1) * d_i) + xp.cos(2.0 * xp.arange(0, N) * d_j))
        + 50000.0
    )

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def apply_periodic_halo(arr, top=0, bottom=0, left=0, right=0):
    """Apply periodic (wrap-around) halo padding to an array.

    Parameters
    ----------
    arr : array
        Input array to pad
    top : int
        Number of rows to add at the top (from bottom of array)
    bottom : int
        Number of rows to add at the bottom (from top of array)
    left : int
        Number of columns to add at the left (from right of array)
    right : int
        Number of columns to add at the right (from left of array)
    """
    xp = arr.__array_namespace__()

    # Apply vertical padding
    if top > 0:
        arr = xp.concatenate([arr[-top:, :], arr], axis=0)
    if bottom > 0:
        arr = xp.concatenate([arr, arr[:bottom, :]], axis=0)

    # Apply horizontal padding
    if left > 0:
        arr = xp.concatenate([arr[:, -left:], arr], axis=1)
    if right > 0:
        arr = xp.concatenate([arr, arr[:, :right]], axis=1)

    return arr


def initialize(xp, M, N, dx, dy, a):
    u, v, p = initialize_interior(xp, M, N, dx, dy, a)

    # Apply staggered 1-halo padding
    u = apply_periodic_halo(u, top=1, right=1)
    v = apply_periodic_halo(v, bottom=1, left=1)
    p = apply_periodic_halo(p, bottom=1, right=1)

    return u, v, p


def initialize_2halo(xp, M, N, dx, dy, a):
    u, v, p = initialize_interior(xp, M, N, dx, dy, a)
    return (
        apply_periodic_halo(u, 1, 1, 1, 1),
        apply_periodic_halo(v, 1, 1, 1, 1),
        apply_periodic_halo(p, 1, 1, 1, 1),
    )
