# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np

from gt4py import gtscript
from gt4py import testing as gt_testing

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS


# ---- Identity stencil ----
class TestIdentity(gt_testing.StencilTestSuite):
    """Identity stencil."""

    dtypes = {("field_a",): (np.float64, np.float32)}
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = CPU_BACKENDS
    symbols = dict(field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(field_a):
        with computation(PARALLEL), interval(...):
            field_a = field_a  # TODO need 'pass'

    def validation(field_a, domain=None, origin=None):
        pass


# ---- Copy stencil ----
class TestCopy(gt_testing.StencilTestSuite):
    """Copy stencil."""

    dtypes = (np.float_,)
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = CPU_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b):
        with computation(PARALLEL), interval(...):
            field_b = field_a

    def validation(field_a, field_b, domain=None, origin=None):
        field_b[...] = field_a


class TestAugAssign(gt_testing.StencilTestSuite):
    """Increment by one stencil."""

    dtypes = (np.float_,)
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = CPU_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b):
        with computation(PARALLEL), interval(...):
            field_a += 1.0
            field_a *= 2.0
            field_b -= 1.0
            field_b /= 2.0

    def validation(field_a, field_b, domain=None, origin=None):
        field_a[...] = (field_a[...] + 1.0) * 2.0
        field_b[...] = (field_b[...] - 1.0) / 2.0


# ---- Scale stencil ----
class TestGlobalScale(gt_testing.StencilTestSuite):
    """Scale stencil using a global global_name."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        SCALE_FACTOR=gt_testing.global_name(one_of=(1.0, 1e3, 1e6)),
        field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a):
        from __externals__ import SCALE_FACTOR

        with computation(PARALLEL), interval(...):
            field_a = SCALE_FACTOR * field_a[0, 0, 0]

    def validation(field_a, domain, origin, **kwargs):
        field_a[...] = SCALE_FACTOR * field_a


# ---- Parametric scale stencil -----
class TestParametricScale(gt_testing.StencilTestSuite):
    """Scale stencil using a parameter."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        scale=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(field_a, *, scale):
        with computation(PARALLEL), interval(...):
            field_a = scale * field_a

    def validation(field_a, *, scale, domain, origin, **kwargs):
        field_a[...] = scale * field_a


# --- Parametric-mix stencil ----
class TestParametricMix(gt_testing.StencilTestSuite):
    """Linear combination of input fields using several parameters."""

    dtypes = {
        ("USE_ALPHA",): np.int_,
        ("field_a", "field_b", "field_c"): np.float64,
        ("field_out",): np.float32,
        ("weight", "alpha_factor"): np.float_,
    }
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = list(set(CPU_BACKENDS) & set(INTERNAL_BACKENDS))
    symbols = dict(
        USE_ALPHA=gt_testing.global_name(one_of=(True, False)),
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_c=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_out=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(-10, 10)),
        alpha_factor=gt_testing.parameter(in_range=(-1, 1)),
    )

    def definition(field_a, field_b, field_c, field_out, *, weight, alpha_factor):
        from __externals__ import USE_ALPHA

        with computation(PARALLEL), interval(...):
            if __INLINED(USE_ALPHA):
                factor = alpha_factor
            else:
                factor = 1.0
            field_out = factor * field_a[0, 0, 0] - (1 - factor) * (
                field_b[0, 0, 0] - weight * field_c[0, 0, 0]
            )

    def validation(
        field_a, field_b, field_c, field_out, *, weight, alpha_factor, domain, origin, **kwargs
    ):
        if USE_ALPHA:
            factor = alpha_factor
        else:
            factor = 1.0
        field_out[...] = (factor * field_a[:, :, :]) - (1 - factor) * (
            field_b[:, :, :] - (weight * field_c[:, :, :])
        )


class TestHeatEquation_FTCS_3D(gt_testing.StencilTestSuite):
    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        u=gt_testing.field(in_range=(-10, 10), extent=[(-1, 1), (0, 0), (0, 0)]),
        v=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (-1, 1), (0, 0)]),
        u_new=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (0, 0), (0, 0)]),
        v_new=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (0, 0), (0, 0)]),
        ru=gt_testing.parameter(in_range=(0, 0.5)),
        rv=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, v, u_new, v_new, *, ru, rv):
        with computation(PARALLEL), interval(...):
            u_new = u[0, 0, 0] + ru * (u[1, 0, 0] - 2 * u[0, 0, 0] + u[-1, 0, 0])
            v_new = v[0, 0, 0] + rv * (v[0, 1, 0] - 2 * v[0, 0, 0] + v[0, -1, 0])

    def validation(u, v, u_new, v_new, *, ru, rv, domain, origin, **kwargs):
        u_new[...] = u[1:-1, :, :] + ru * (u[2:, :, :] - 2 * u[1:-1, :, :] + u[:-2, :, :])
        v_new[...] = v[:, 1:-1, :] + rv * (v[:, 2:, :] - 2 * v[:, 1:-1, :] + v[:, :-2, :])


class TestHorizontalDiffusion(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        with computation(PARALLEL), interval(...):
            laplacian = 4.0 * u[0, 0, 0] - (u[1, 0, 0] + u[-1, 0, 0] + u[0, 1, 0] + u[0, -1, 0])
            flux_i = laplacian[1, 0, 0] - laplacian[0, 0, 0]
            flux_j = laplacian[0, 1, 0] - laplacian[0, 0, 0]
            diffusion = u[0, 0, 0] - weight * (
                flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0]
            )

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


@gtscript.function
def lap_op(u):
    """Laplacian operator."""
    return 4.0 * u[0, 0, 0] - (u[1, 0, 0] + u[-1, 0, 0] + u[0, 1, 0] + u[0, -1, 0])


@gtscript.function
def fwd_diff_op_xy(field):
    dx = field[1, 0, 0] - field[0, 0, 0]
    dy = field[0, 1, 0] - field[0, 0, 0]
    return dx, dy


@gtscript.function
def wrap1arg2return(field):
    dx, dy = fwd_diff_op_xy(field=field)
    return dx, dy


@gtscript.function
def fwd_diff_op_x(field):
    dx = field[1, 0, 0] - field[0, 0, 0]
    return dx


@gtscript.function
def fwd_diff_op_y(field):
    dy = field[0, 1, 0] - field[0, 0, 0]
    return dy


class TestHorizontalDiffusionSubroutines(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        fwd_diff=gt_testing.global_name(singleton=wrap1arg2return),
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        from __externals__ import fwd_diff

        with computation(PARALLEL), interval(...):
            laplacian = lap_op(u=u)
            flux_i, flux_j = fwd_diff(field=laplacian)
            diffusion = u[0, 0, 0] - weight * (
                flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0]
            )

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


class TestHorizontalDiffusionSubroutines2(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        fwd_diff=gt_testing.global_name(singleton=fwd_diff_op_xy),
        BRANCH=gt_testing.global_name(one_of=(True, False)),
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        from __externals__ import BRANCH, fwd_diff

        with computation(PARALLEL), interval(...):
            laplacian = lap_op(u=u)
            if __INLINED(BRANCH):
                flux_i = fwd_diff_op_x(field=laplacian)
                flux_j = fwd_diff_op_y(field=laplacian)
            else:
                flux_i, flux_j = fwd_diff_op_xy(field=laplacian)
            diffusion = u[0, 0, 0] - weight * (
                flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0]
            )

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


class TestRuntimeIfFlat(gt_testing.StencilTestSuite):
    """Tests runtime ifs."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)])
    )

    def definition(outfield):

        with computation(PARALLEL), interval(...):

            if 1:
                outfield = 1
            else:
                outfield = 2

    def validation(outfield, *, domain, origin, **kwargs):
        outfield[...] = 1


class TestRuntimeIfNested(gt_testing.StencilTestSuite):
    """Tests nested runtime ifs"""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)])
    )

    def definition(outfield):

        with computation(PARALLEL), interval(...):
            if (outfield and outfield) or (not outfield and not outfield):
                if 0:
                    outfield = 1
                else:
                    outfield = 2
            else:
                outfield = 3

    def validation(outfield, *, domain, origin, **kwargs):
        outfield[...] = 2


@gtscript.function
def add_one(field_in):
    """Add 1 to each element of `field_in`."""
    return field_in + 1


class Test3FoldNestedIf(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(3, 3), (3, 3), (3, 3)]
    backends = ["debug", "numpy", "gtx86"]
    symbols = dict(field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(field_a):
        with computation(PARALLEL), interval(...):
            if field_a >= 0.0:
                field_a = 0.0
                if field_a > 1:
                    field_a = 1
                    if field_a > 2:
                        field_a = 2

    def validation(field_a, domain, origin):
        for v in range(3):
            field_a[np.where(field_a > v)] = v


class TestRuntimeIfNestedDataDependent(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(3, 3), (3, 3), (3, 3)]
    backends = CPU_BACKENDS
    symbols = dict(
        # factor=gt_testing.global_name(one_of=(-1., 0., 1.)),
        factor=gt_testing.parameter(in_range=(-100, 100)),
        field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_c=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b, field_c, *, factor):
        # from __externals__ import factor
        with computation(PARALLEL), interval(...):
            if factor > 0:
                if field_a < 0:
                    field_b = -field_a
                else:
                    field_b = field_a
            else:
                if field_a < 0:
                    field_c = -field_a
                else:
                    field_c = field_a

            field_a = add_one(field_a)

    def validation(field_a, field_b, field_c, *, factor, domain, origin, **kwargs):

        if factor > 0:
            field_b[...] = np.abs(field_a)
        else:
            field_c[...] = np.abs(field_a)
        field_a += 1


class TestTernaryOp(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        infield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 1), (0, 0)]),
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(infield, outfield):

        with computation(PARALLEL), interval(...):
            outfield = (infield > 0.0) * infield + (infield <= 0.0) * (-infield[0, 1, 0])

    def validation(infield, outfield, *, domain, origin, **kwargs):
        outfield[...] = np.choose(infield[:, :-1, :] > 0, [-infield[:, 1:, :], infield[:, :-1, :]])


class TestThreeWayAnd(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        a=gt_testing.parameter(in_range=(-100, 100)),
        b=gt_testing.parameter(in_range=(-100, 100)),
        c=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(outfield, *, a, b, c):

        with computation(PARALLEL), interval(...):
            if a > 0 and b > 0 and c > 0:
                outfield = 1
            else:
                outfield = 0

    def validation(outfield, *, a, b, c, domain, origin, **kwargs):
        outfield[...] = 1 if a > 0 and b > 0 and c > 0 else 0


class TestThreeWayOr(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = CPU_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        a=gt_testing.parameter(in_range=(-100, 100)),
        b=gt_testing.parameter(in_range=(-100, 100)),
        c=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(outfield, *, a, b, c):

        with computation(PARALLEL), interval(...):
            if a > 0 or b > 0 or c > 0:
                outfield = 1
            else:
                outfield = 0

    def validation(outfield, *, a, b, c, domain, origin, **kwargs):
        outfield[...] = 1 if a > 0 or b > 0 or c > 0 else 0
