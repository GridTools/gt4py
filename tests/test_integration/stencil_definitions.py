# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import gt4py as gt
from gt4py import analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import ir as gt_ir
from gt4py import storage as gt_storage
from gt4py import utils as gt_utils
from gt4py.definitions import Extent, StencilID


REGISTRY = gt_utils.Registry()
EXTERNALS_REGISTRY = gt_utils.Registry()


def register(func=None, *, externals=None, name=None):
    def _register_decorator(actual_func):
        EXTERNALS_REGISTRY.register(name or actual_func.__name__, externals or {})
        REGISTRY.register(name or actual_func.__name__, actual_func)
        return actual_func

    return _register_decorator(func) if func else _register_decorator


Field0D = gtscript.Field[np.float_, ()]
Field3D = gtscript.Field[np.float_]
Field3DBool = gtscript.Field[np.bool]


@register
def copy_stencil(field_a: Field3D, field_b: Field3D):
    with computation(PARALLEL), interval(...):
        field_b = field_a[0, 0, 0]


@gtscript.function
def afunc(b):
    return sqrt(b[0, 1, 0])


@register
def native_functions(field_a: Field3D, field_b: Field3D):
    with computation(PARALLEL), interval(...):
        field_a = max(min(afunc(field_b), field_b), 1)


@register
def copy_stencil_plus_one(field_a: Field3D, field_b: Field3D):
    with computation(PARALLEL), interval(...):
        field_b = field_a[0, 0, 0] + 1


@register
def runtime_if(field_a: Field3D, field_b: Field3D):
    with computation(BACKWARD), interval(...):
        if field_a > 0.0:
            field_b = -1
            field_a = -field_a
        else:
            field_b = 1
            field_a = field_a


@register
def simple_horizontal_diffusion(in_field: Field3D, coeff: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )
        flx_field = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        fly_field = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )


@register
def tridiagonal_solver(inf: Field3D, diag: Field3D, sup: Field3D, rhs: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            sup = sup / diag
            rhs = rhs / diag
        with interval(1, None):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
    with computation(BACKWARD):
        with interval(-1, None):
            out = rhs
        with interval(0, -1):
            out = rhs - sup * out[0, 0, 1]


@register(externals={"BET_M": 0.5, "BET_P": 0.5})
def vertical_advection_dycore(
    utens_stage: Field3D,
    u_stage: Field3D,
    wcon: Field3D,
    u_pos: Field3D,
    utens: Field3D,
    *,
    dtr_stage: float,
):
    from __externals__ import BET_M, BET_P

    with computation(FORWARD):
        with interval(0, 1):
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
            cs = gcv * BET_M

            ccol = gcv * BET_P
            bcol = dtr_stage - ccol[0, 0, 0]

            # update the d column
            correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term
            )

            # Thomas forward
            divided = 1.0 / bcol[0, 0, 0]
            ccol = ccol[0, 0, 0] * divided
            dcol = dcol[0, 0, 0] * divided

        with interval(1, -1):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

            as_ = gav * BET_M
            cs = gcv * BET_M

            acol = gav * BET_P
            ccol = gcv * BET_P
            bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (
                u_stage[0, 0, 1] - u_stage[0, 0, 0]
            )
            dcol = (
                dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            ccol = ccol[0, 0, 0] * divided
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with interval(-1, None):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            as_ = gav * BET_M
            acol = gav * BET_P
            bcol = dtr_stage - acol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

    with computation(BACKWARD):
        with interval(-1, None):
            datacol = dcol[0, 0, 0]
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

        with interval(0, -1):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * datacol[0, 0, 1]
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])


@register
def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )


@register
def large_k_interval(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL):
        with interval(0, 6):
            out_field = in_field
        with interval(6, -10):  # this stage will only run if field has more than 16 elements
            out_field = in_field + 1
        with interval(-10, None):
            out_field = in_field


@register
def form_land_mask(in_field: Field3D, mask: gtscript.Field[np.bool]):
    with computation(PARALLEL), interval(...):
        mask = in_field >= 0


@register
def set_inner_as_kord(a4_1: Field3D, a4_2: Field3D, a4_3: Field3D, extm: Field3DBool):
    with computation(PARALLEL), interval(...):
        diff_23 = 0.0
        if extm and extm[0, 0, -1]:
            a4_2 = a4_1
        elif extm and extm[0, 0, 1]:
            a4_3 = a4_1
        else:
            diff_23 = a4_2 - a4_3


@register
def local_var_inside_nested_conditional(in_storage: Field3D, out_storage: Field3D):
    with computation(PARALLEL), interval(0, 2):
        mid_storage = 2
        if in_storage[0, 0, 0] > 0:
            local_var = 4
            if local_var + in_storage < out_storage:
                mid_storage = 3
            else:
                mid_storage = 4
            out_storage[0, 0, 0] = local_var + mid_storage
    with computation(FORWARD), interval(2, None):
        if in_storage[0, 0, 0] < 0:
            local_var = 6
            out_storage[0, 0, 0] = local_var + mid_storage


@register
def multibranch_param_conditional(in_field: Field3D, out_field: Field3D, c: float):
    with computation(PARALLEL), interval(...):
        if c > 0.0:
            out_field = in_field + in_field[1, 0, 0]
        elif c < -1.0:
            out_field = in_field - in_field[1, 0, 0]
        else:
            out_field = in_field


@register(externals={"DO_SOMETHING": False})
def allow_empty_computation(in_field: Field3D, out_field: Field3D):
    from __externals__ import DO_SOMETHING

    with computation(FORWARD), interval(...):
        out_field = in_field
    with computation(PARALLEL), interval(...):
        if __INLINED(DO_SOMETHING):
            out_field = abs(in_field)


@register(externals={"PHYS_TEND": False}, name="unused_optional_field")
@register(externals={"PHYS_TEND": True}, name="required_optional_field")
def optional_field(
    in_field: Field3D,
    out_field: Field3D,
    dyn_tend: Field3D,
    phys_tend: Field3D = None,
    *,
    dt: float,
):
    from __externals__ import PHYS_TEND

    with computation(PARALLEL), interval(...):
        out_field = in_field + dt * dyn_tend
        if __INLINED(PHYS_TEND):
            out_field = out_field + dt * phys_tend


@register(externals={"PHYS_TEND_A": False, "PHYS_TEND_B": False}, name="two_optional_fields_00")
@register(externals={"PHYS_TEND_A": False, "PHYS_TEND_B": True}, name="two_optional_fields_01")
@register(externals={"PHYS_TEND_A": True, "PHYS_TEND_B": True}, name="two_optional_fields_11")
def two_optional_fields(
    in_a: Field3D,
    in_b: Field3D,
    out_a: Field3D,
    out_b: Field3D,
    dyn_tend_a: Field3D,
    dyn_tend_b: Field3D,
    phys_tend_a: Field3D = None,
    phys_tend_b: Field3D = None,
    *,
    dt: float,
):
    from __externals__ import PHYS_TEND_A, PHYS_TEND_B

    with computation(PARALLEL), interval(...):
        out_a = in_a + dt * dyn_tend_a
        out_b = in_b + dt * dyn_tend_b
        if __INLINED(PHYS_TEND_A):
            out_a = out_a + dt * phys_tend_a
        if __INLINED(PHYS_TEND_B):
            out_b = out_b + dt * phys_tend_b
