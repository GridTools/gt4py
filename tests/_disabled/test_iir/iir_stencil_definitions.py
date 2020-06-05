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


import gt4py as gt
from gt4py import ir as gt_ir
from gt4py.definitions import Extent, StencilID
import gt4py.backend as gt_backend
import gt4py.utils as gt_utils

REGISTRY = gt_utils.Registry()


def register(iir_factory):
    if iir_factory.__name__.startswith("make_"):
        name = iir_factory.__name__[len("make_") :]
    else:
        raise ValueError("Name of stencil factory must start with 'make_'")
    return REGISTRY.register(name, iir_factory)


def build_iir_stencil(name, options, backend="gtx86", *, id_version="xxxxxx"):
    if isinstance(backend, str):
        backend = gt_backend.from_name(backend)
    if not issubclass(backend, gt_backend.Backend):
        raise TypeError("Backend must be a backend identifier string or a gt4py Backend class.")

    iir_factory = REGISTRY[name]
    iir = iir_factory()
    stencil_id = StencilID("{}.{}".format(options.module, options.name), id_version)

    if options.rebuild:
        # Force recompilation
        stencil_class = None
    else:
        # Use cached version (if id_version matches)
        stencil_class = backend.load(stencil_id, None, options)

    if stencil_class is None:
        stencil_class = backend.generate(stencil_id, iir, None, options)

    stencil_implementation = stencil_class()

    return stencil_implementation


def dump_all_to_file():
    for name in REGISTRY.names:
        iir_factory = REGISTRY[name]
        iir = iir_factory()
        with open(name + ".json", "w") as f:
            f.write(gt_ir.utils.dump_ir(iir, as_json=True))


@register
def make_copy_stencil():
    def stage(field_a, field_b):
        field_b = field_a[0, 0, 0]

    stage_1 = gt_ir.utils.make_stage(
        stage,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "field_a": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "field_b": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    multi_stage_1 = gt_ir.utils.make_multi_stage(
        "multi_stage_1", gt_ir.IterationOrder.PARALLEL, [stage_1]
    )

    implementation = gt_ir.utils.make_implementation(
        "copy_stencil",
        args_list=["field_a", "field_b"],
        fields_with_description={
            "field_a": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "field_b": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[multi_stage_1],
    )

    return implementation


@register
def make_copy_stencil_plus_one():
    def stage(field_a, field_b):
        field_b = field_a[0, 0, 0] + 1.0

    stage_1 = gt_ir.utils.make_stage(
        stage,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "field_a": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "field_b": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    multi_stage_1 = gt_ir.utils.make_multi_stage(
        "multi_stage_1", gt_ir.IterationOrder.PARALLEL, [stage_1]
    )

    implementation = gt_ir.utils.make_implementation(
        "copy_stencil",
        args_list=["field_a", "field_b"],
        fields_with_description={
            "field_a": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "field_b": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[multi_stage_1],
    )

    return implementation


@register
def make_simple_horizontal_diffusion():
    # Laplacian
    def laplacian(in_field, lap_field):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )

    laplacian_stage = gt_ir.utils.make_stage(
        laplacian,
        compute_extent=[(-1, 1), (-1, 1), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(-1, 1), (-1, 1), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    # Flux X
    def flx(in_field, lap_field, flx_field):
        flx_field = lap_field[1, 0, 0] - lap_field[0, 0, 0]

    flx_stage = gt_ir.utils.make_stage(
        flx,
        compute_extent=[(-1, 0), (0, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 0)]),
            "flx_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    # Flux Y
    def fly(in_field, lap_field, fly_field):
        fly_field = lap_field[0, 1, 0] - lap_field[0, 0, 0]

    fly_stage = gt_ir.utils.make_stage(
        fly,
        compute_extent=[(0, 0), (-1, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 1), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 1), (0, 0)]),
            "fly_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            # res=float,
        },
    )

    # Out
    def out(in_field, flx_field, fly_field, coeff, out_field):
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )

    out_stage = gt_ir.utils.make_stage(
        out,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "flx_field": (gt_ir.AccessIntent.READ_ONLY, [(-1, 0), (0, 0), (0, 0)]),
            "fly_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (-1, 0), (0, 0)]),
            "coeff": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "out_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    multi_stage = gt_ir.utils.make_multi_stage(
        "multi_stage_1",
        gt_ir.IterationOrder.PARALLEL,
        [laplacian_stage, [flx_stage, fly_stage], out_stage],
    )

    implementation = gt_ir.utils.make_implementation(
        "horizontal_diffusion",
        args_list=["in_field", "coeff", "out_field"],
        fields_with_description={
            "in_field": dict(is_api=True, extent=[(-2, 2), (-2, 2), (0, 0)]),
            "coeff": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "lap_field": dict(is_api=False, extent=[(-1, 1), (-1, 1), (0, 0)]),
            "flx_field": dict(is_api=False, extent=[(-1, 1), (0, 0), (0, 0)]),
            "fly_field": dict(is_api=False, extent=[(0, 0), (-1, 1), (0, 0)]),
            "out_field": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[multi_stage],
    )

    return implementation


@register
def make_horizontal_diffusion():
    # Laplacian
    def laplacian(in_field, lap_field):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )

    laplacian_stage = gt_ir.utils.make_stage(
        laplacian,
        compute_extent=[(-1, 1), (-1, 1), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(-1, 1), (-1, 1), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    # Flux X
    def flx(in_field, lap_field, flx_field):
        # flx_field = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res

    flx_stage = gt_ir.utils.make_stage(
        flx,
        compute_extent=[(-1, 0), (0, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 0)]),
            "flx_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
        local_symbols={"res": float},
    )

    # Flux Y
    def fly(in_field, lap_field, fly_field):
        # fly_field = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res

    fly_stage = gt_ir.utils.make_stage(
        fly,
        compute_extent=[(0, 0), (-1, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 1), (0, 0)]),
            "lap_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 1), (0, 0)]),
            "fly_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            # res=float,
        },
        local_symbols={"res": float},
    )

    # Out
    def out(in_field, flx_field, fly_field, coeff, out_field):
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )

    out_stage = gt_ir.utils.make_stage(
        out,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "in_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "flx_field": (gt_ir.AccessIntent.READ_ONLY, [(-1, 0), (0, 0), (0, 0)]),
            "fly_field": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (-1, 0), (0, 0)]),
            "coeff": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "out_field": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    multi_stage = gt_ir.utils.make_multi_stage(
        "multi_stage_1",
        gt_ir.IterationOrder.PARALLEL,
        [laplacian_stage, [flx_stage, fly_stage], out_stage],
    )

    implementation = gt_ir.utils.make_implementation(
        "horizontal_diffusion",
        args_list=["in_field", "coeff", "out_field"],
        fields_with_description={
            "in_field": dict(is_api=True, extent=[(-2, 2), (-2, 2), (0, 0)]),
            "coeff": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "lap_field": dict(is_api=False, extent=[(-1, 1), (-1, 1), (0, 0)]),
            "flx_field": dict(is_api=False, extent=[(-1, 0), (0, 0), (0, 0)]),
            "fly_field": dict(is_api=False, extent=[(0, 0), (-1, 0), (0, 0)]),
            "out_field": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[multi_stage],
    )
    # from ..test_def_ir.test_ir_transformation import analyze
    # ref_iir = analyze("horizontal_diffusion")
    return implementation


@register
def make_test_vertical_regions():
    def set_stage_id(out):
        with gt.region(iteration=gt.PARALLEL, k_interval=(0, 1)):
            out = 1

        with gt.region(iteration=gt.PARALLEL, k_interval=(1, -1)):
            out = 2

        with gt.region(iteration=gt.PARALLEL, k_interval=(-1, None)):
            out = 3

    stage = gt_ir.utils.make_stage(
        set_stage_id,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={"out": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)])},
        parameters={},
        local_symbols={},
    )

    multi_stage = gt_ir.utils.make_multi_stage(
        "multi_stage", gt_ir.IterationOrder.PARALLEL, [stage]
    )

    implementation = gt_ir.utils.make_implementation(
        "test_vertical_regions",
        args_list=["out"],
        fields_with_description={"out": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)])},
        parameters_with_type={},
        multi_stages=[multi_stage],
    )

    return implementation


@register
def make_tridiagonal_solver():
    # Forward
    def forward_thomas(inf, diag, sup, rhs, out):
        with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
            sup = sup / diag
            rhs = rhs / diag
        with gt.region(iteration=gt.FORWARD, k_interval=(1, None)):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)

    forward_stage = gt_ir.utils.make_stage(
        forward_thomas,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "inf": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "diag": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "sup": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
            "rhs": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
            "out": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    # Backward
    def backward_thomas(inf, diag, sup, rhs, out):
        with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
            out = rhs - sup * out[0, 0, 1]
        with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
            out = rhs

    backward_stage = gt_ir.utils.make_stage(
        backward_thomas,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "inf": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "diag": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "sup": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "rhs": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "out": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 1)]),
        },
    )

    forward_multi_stage = gt_ir.utils.make_multi_stage(
        "forward_ms", gt_ir.IterationOrder.FORWARD, [forward_stage]
    )

    backward_multi_stage = gt_ir.utils.make_multi_stage(
        "backward_ms", gt_ir.IterationOrder.BACKWARD, [backward_stage]
    )

    implementation = gt_ir.utils.make_implementation(
        "tridiagonal_solver",
        args_list=["inf", "diag", "sup", "rhs", "out"],
        fields_with_description={
            "inf": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "diag": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "sup": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "rhs": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "out": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[forward_multi_stage, backward_multi_stage],
    )

    return implementation


@register
def make_vertical_advection_dycore():
    BETA_V = 0.0
    BET_M = (1 - BETA_V) / 2.0
    BET_P = (1 + BETA_V) / 2.0

    def u_forward_function(
        utens_stage, wcon, u_stage, u_pos, utens, acol, bcol, ccol, dcol, *, dtr_stage
    ):
        with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
            cs = gcv * BET_M

            ccol = gcv * BET_P
            bcol = dtr_stage - ccol[0, 0, 0]

            # update the d column
            correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / bcol[0, 0, 0]
            ccol = ccol[0, 0, 0] * divided
            dcol = dcol[0, 0, 0] * divided

        with gt.region(iteration=gt.FORWARD, k_interval=(1, -1)):
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
                dtr_stage * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            ccol = ccol[0, 0, 0] * divided
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with gt.region(iteration=gt.FORWARD, k_interval=(-1, None)):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            as_ = gav * BET_M
            acol = gav * BET_P
            bcol = dtr_stage - acol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

    forward_stage = gt_ir.utils.make_stage(
        u_forward_function,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "utens_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "wcon": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 1)]),
            "u_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (-1, 1)]),
            "u_pos": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "utens": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "acol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "bcol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "ccol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
            "dcol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
        },
        parameters={"dtr_stage": float},
        local_symbols={
            "gav": float,
            "gcv": float,
            "as_": float,
            "cs": float,
            "correction_term": float,
            "divided": float,
        },
        externals={"BETA_V": BETA_V, "BET_M": BET_M, "BET_P": BET_P},
    )

    def u_backward_function(utens_stage, u_pos, ccol, dcol, data_col, *, dtr_stage):
        with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

        with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
            datacol = dcol[0, 0, 0]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

    backward_stage = gt_ir.utils.make_stage(
        u_backward_function,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "utens_stage": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "u_pos": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "ccol": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "dcol": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "data_col": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 1)]),
        },
        parameters={"dtr_stage": float},
        local_symbols={"datacol": float},
        externals={"BETA_V": BETA_V, "BET_M": BET_M, "BET_P": BET_P},
    )

    forward_multi_stage = gt_ir.utils.make_multi_stage(
        "forward_ms", gt_ir.IterationOrder.FORWARD, [forward_stage]
    )

    backward_multi_stage = gt_ir.utils.make_multi_stage(
        "backward_ms", gt_ir.IterationOrder.BACKWARD, [backward_stage]
    )

    implementation = gt_ir.utils.make_implementation(
        "vertical_advection_dycore",
        args_list=[
            "utens_stage",
            "u_stage",
            "wcon",
            "u_pos",
            "utens",
            # gt_ir.ArgumentInfo(name="dtr_stage", is_keyword=True),
            "dtr_stage",
        ],
        fields_with_description={
            "utens_stage": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "u_stage": dict(is_api=True, extent=[(0, 0), (0, 0), (-1, 1)]),
            "wcon": dict(is_api=True, extent=[(0, 1), (0, 0), (0, 1)]),
            "u_pos": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "utens": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "acol": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 0)]),
            "bcol": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 0)]),
            "ccol": dict(is_api=False, extent=[(0, 0), (0, 0), (-1, 0)]),
            "dcol": dict(is_api=False, extent=[(0, 0), (0, 0), (-1, 0)]),
            "data_col": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 1)]),
        },
        parameters_with_type={"dtr_stage": float},
        multi_stages=[forward_multi_stage, backward_multi_stage],
    )

    return implementation


@register
def make_vertical_advection_dycore_with_scalar_storage():
    BETA_V = 0.0
    BET_M = (1 - BETA_V) / 2.0
    BET_P = (1 + BETA_V) / 2.0

    def u_forward_function(
        utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol
    ):
        with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
            cs = gcv * BET_M

            ccol = gcv * BET_P
            bcol = dtr_stage[0, 0, 0] - ccol[0, 0, 0]

            # update the d column
            correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage[0, 0, 0] * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / bcol[0, 0, 0]
            ccol = ccol[0, 0, 0] * divided
            dcol = dcol[0, 0, 0] * divided

        with gt.region(iteration=gt.FORWARD, k_interval=(1, -1)):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

            as_ = gav * BET_M
            cs = gcv * BET_M

            acol = gav * BET_P
            ccol = gcv * BET_P
            bcol = dtr_stage[0, 0, 0] - acol[0, 0, 0] - ccol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (
                u_stage[0, 0, 1] - u_stage[0, 0, 0]
            )
            dcol = (
                dtr_stage[0, 0, 0] * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            ccol = ccol[0, 0, 0] * divided
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with gt.region(iteration=gt.FORWARD, k_interval=(-1, None)):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            as_ = gav * BET_M
            acol = gav * BET_P
            bcol = dtr_stage[0, 0, 0] - acol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
            dcol = (
                dtr_stage[0, 0, 0] * u_pos[0, 0, 0]
                + utens[0, 0, 0]
                + utens_stage[0, 0, 0]
                + correction_term
            )

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

    forward_stage = gt_ir.utils.make_stage(
        u_forward_function,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "utens_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "wcon": (gt_ir.AccessIntent.READ_ONLY, [(0, 1), (0, 0), (0, 1)]),
            "u_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (-1, 1)]),
            "u_pos": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "utens": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "dtr_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "acol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "bcol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "ccol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
            "dcol": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (-1, 0)]),
        },
        parameters={},
        local_symbols={
            "gav": float,
            "gcv": float,
            "as_": float,
            "cs": float,
            "correction_term": float,
            "divided": float,
        },
        externals={"BETA_V": BETA_V, "BET_M": BET_M, "BET_P": BET_P},
    )

    def u_backward_function(utens_stage, u_pos, dtr_stage, ccol, dcol, data_col):
        with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
            data_col = datacol
            utens_stage = dtr_stage[0, 0, 0] * (datacol - u_pos[0, 0, 0])

        with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
            datacol = dcol[0, 0, 0]
            data_col = datacol
            utens_stage = dtr_stage[0, 0, 0] * (datacol - u_pos[0, 0, 0])

    backward_stage = gt_ir.utils.make_stage(
        u_backward_function,
        compute_extent=[(0, 0), (0, 0), (0, 0)],
        fields_with_access={
            "utens_stage": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
            "u_pos": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "dtr_stage": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "ccol": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "dcol": (gt_ir.AccessIntent.READ_ONLY, [(0, 0), (0, 0), (0, 0)]),
            "data_col": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 1)]),
        },
        parameters={},
        local_symbols={"datacol": float},
        externals={"BETA_V": BETA_V, "BET_M": BET_M, "BET_P": BET_P},
    )

    forward_multi_stage = gt_ir.utils.make_multi_stage(
        "forward_ms", gt_ir.IterationOrder.FORWARD, [forward_stage]
    )

    backward_multi_stage = gt_ir.utils.make_multi_stage(
        "backward_ms", gt_ir.IterationOrder.BACKWARD, [backward_stage]
    )

    implementation = gt_ir.utils.make_implementation(
        "vertical_advection_dycore",
        args_list=["utens_stage", "u_stage", "wcon", "u_pos", "utens", "dtr_stage"],
        fields_with_description={
            "utens_stage": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "u_stage": dict(is_api=True, extent=[(0, 0), (0, 0), (-1, 1)]),
            "wcon": dict(is_api=True, extent=[(0, 1), (0, 0), (0, 1)]),
            "u_pos": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "utens": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "dtr_stage": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
            "acol": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 0)]),
            "bcol": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 0)]),
            "ccol": dict(is_api=False, extent=[(0, 0), (0, 0), (-1, 0)]),
            "dcol": dict(is_api=False, extent=[(0, 0), (0, 0), (-1, 0)]),
            "data_col": dict(is_api=False, extent=[(0, 0), (0, 0), (0, 1)]),
        },
        parameters_with_type={},
        multi_stages=[forward_multi_stage, backward_multi_stage],
    )

    return implementation


if __name__ == "__main__":
    dump_all_to_file()
