# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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
import gt4py.analysis as gt_analysis
from gt4py import storage as gt_storage
import gt4py.gtscript as gtscript
import gt4py.utils as gt_utils

REGISTRY = gt_utils.Registry()


def register(def_ir_factory):
    if def_ir_factory.__name__.startswith("make_"):
        name = def_ir_factory.__name__[len("make_") :]
    else:
        raise ValueError("Name of stencil factory must start with 'make_'")
    return REGISTRY.register(name, def_ir_factory)


def dump_all_to_file():
    for name in REGISTRY.names:
        def_ir_factory = REGISTRY[name]
        iir = def_ir_factory()
        with open(name + "_def_IR.json", "w") as f:
            f.write(gt_ir.utils.dump_ir(iir, as_json=True))


def build_def_ir_stencil(name, options, backend="gtx86", *, id_version="xxxxxx"):
    if isinstance(backend, str):
        backend = gt_backend.from_name(backend)
    if not issubclass(backend, gt_backend.Backend):
        raise TypeError("Backend must be a backend identifier string or a gt4py Backend class.")

    def_ir_factory = REGISTRY[name]
    def_ir = def_ir_factory()
    iir = gt_analysis.transform(def_ir, options)
    import tests.test_iir.iir_stencil_definitions as iir_stencil_definitions

    ref_iir = iir_stencil_definitions.make_vertical_advection_dycore()
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


@register
def make_copy_stencil():
    def definition_func(field_a, field_b):
        field_b = field_a[0, 0, 0]

    definition = gt_ir.utils.make_definition(
        "copy_stencil",
        definition_func,
        args_list=["field_a", "field_b"],
        fields_with_storage_descriptor={
            "field_a": gtscript.Field[float],
            "field_b": gtscript.Field[float],
        },
        parameters_with_type={},
        temp_fields_with_type={},
    )

    return definition


@register
def make_copy_stencil_plus_one():
    def definition_func(field_a, field_b):
        field_b = field_a[0, 0, 0] + 1

    definition = gt_ir.utils.make_definition(
        "copy_stencil_plus_one",
        definition_func,
        args_list=["field_a", "field_b"],
        fields_with_storage_descriptor={
            "field_a": gtscript.Field[float],
            "field_b": gtscript.Field[float],
        },
        parameters_with_type={},
        temp_fields_with_type={},
    )

    return definition


@register
def make_simple_horizontal_diffusion():
    def definition_func(in_field, coeff, out_field):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )
        flx_field = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        fly_field = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )

    definition = gt_ir.utils.make_definition(
        "simple_horizontal_diffusion",
        definition_func,
        args_list=["in_field", "out_field", "coeff"],
        fields_with_storage_descriptor={
            "in_field": gtscript.Field[float],
            "coeff": gtscript.Field[float],
            "out_field": gtscript.Field[float],
        },
        parameters_with_type={},
        temp_fields_with_type={"lap_field": float, "flx_field": float, "fly_field": float},
    )

    return definition


@register
def make_tridiagonal_solver():
    def definition_func(inf, diag, sup, rhs, out):
        with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
            sup = sup / diag
            rhs = rhs / diag
        with gt.region(iteration=gt.FORWARD, k_interval=(1, None)):
            sup = sup / (diag - sup[0, 0, -1] * inf)
            rhs = (rhs - inf * rhs[0, 0, -1]) / (diag - sup[0, 0, -1] * inf)
        with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
            out = rhs - sup * out[0, 0, 1]
        with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
            out = rhs

    definition = gt_ir.utils.make_definition(
        "tridiagonal_solver",
        definition_func,
        args_list=["inf", "diag", "sup", "rhs", "out"],
        fields_with_storage_descriptor={
            "inf": gtscript.Field[float],
            "diag": gtscript.Field[float],
            "sup": gtscript.Field[float],
            "rhs": gtscript.Field[float],
            "out": gtscript.Field[float],
        },
        temp_fields_with_type={},
        parameters_with_type={},
    )

    return definition


@register
def make_vertical_advection_dycore():
    BETA_V = 0.0
    BET_M = (1 - BETA_V) / 2.0
    BET_P = (1 + BETA_V) / 2.0

    def definition_func(utens_stage, u_stage, wcon, u_pos, utens, *, dtr_stage):
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
        # with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
        #     datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
        #     data_col = datacol
        #     utens_stage = dtr_stage * (datacol) - u_pos[0, 0, 0]

        with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
            datacol = dcol[0, 0, 0]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])
        with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

    definition = gt_ir.utils.make_definition(
        "vertical_advection_dycore",
        definition_func,
        args_list=["utens_stage", "u_stage", "wcon", "u_pos", "utens", "dtr_stage"],
        fields_with_storage_descriptor={
            "utens_stage": gtscript.Field[float],
            "u_stage": gtscript.Field[float],
            "wcon": gtscript.Field[float],
            "u_pos": gtscript.Field[float],
            "utens": gtscript.Field[float],
        },
        temp_fields_with_type={
            "acol": float,
            "bcol": float,
            "ccol": float,
            "dcol": float,
            "data_col": float,
            "gav": float,
            "gcv": float,
            "cs": float,
            "correction_term": float,
            "divided": float,
            "as_": float,
            "datacol": float,
        },
        parameters_with_type={"dtr_stage": float},
        # parameters_with_type={
        #     "dtr_stage": float,
        #     "gav": float,
        #     "gcv": float,
        #     "as_": float,
        #     "cs": float,
        #     "correction_term": float,
        #     "divided": float,
        #     "datacol": float,
        # },
        externals={"BET_M": BET_M, "BET_P": BET_P},
    )

    return definition


@register
def make_horizontal_diffusion():
    # Laplacian
    def definition_func(in_field, out_field, coeff):
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

    implementation = gt_ir.utils.make_definition(
        "horizontal_diffusion",
        definition_func,
        args_list=["in_field", "out_field", "coeff"],
        fields_with_storage_descriptor={
            "in_field": gtscript.Field[float],
            "out_field": gtscript.Field[float],
            "coeff": gtscript.Field[float],
        },
        temp_fields_with_type={
            "lap_field": float,
            "flx_field": float,
            "fly_field": float,
            "res": float,
        },
        parameters_with_type={},
    )

    return implementation


if __name__ == "__main__":
    dump_all_to_file()
