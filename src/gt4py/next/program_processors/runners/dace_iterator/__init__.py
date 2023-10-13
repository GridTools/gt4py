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
import hashlib
from typing import Any, Mapping, Optional, Sequence

import dace
import numpy as np
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.iterator.ir as itir
from gt4py.next.common import Dimension, Domain, UnitRange, is_field
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider, StridedNeighborOffsetProvider
from gt4py.next.iterator.transforms import LiftMode, apply_common_transforms, global_tmps
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.type_system import type_specifications as ts, type_translation

from .itir_to_sdfg import ItirToSDFG
from .utility import connectivity_identifier, filter_neighbor_tables, get_sorted_dims


def get_sorted_dim_ranges(domain: Domain) -> Sequence[UnitRange]:
    sorted_dims = get_sorted_dims(domain.dims)
    return [domain.ranges[dim_index] for dim_index, _ in sorted_dims]


""" Default build configuration in DaCe backend """
_build_type = "Release"
# removing  -ffast-math from DaCe default compiler args in order to support isfinite/isinf/isnan built-ins
_cpu_args = (
    "-std=c++14 -fPIC -Wall -Wextra -O3 -march=native -Wno-unused-parameter -Wno-unused-label"
)


def convert_arg(arg: Any):
    if is_field(arg):
        sorted_dims = get_sorted_dims(arg.domain.dims)
        ndim = len(sorted_dims)
        dim_indices = [dim_index for dim_index, _ in sorted_dims]
        assert isinstance(arg.ndarray, np.ndarray)
        return np.moveaxis(arg.ndarray, range(ndim), dim_indices)
    return arg


def preprocess_program(
    program: itir.FencilDefinition, offset_provider: Mapping[str, Any], lift_mode: LiftMode
):
    program = apply_common_transforms(
        program,
        offset_provider=offset_provider,
        lift_mode=lift_mode,
        common_subexpression_elimination=False,
    )
    return program


def get_args(params: Sequence[itir.Sym], args: Sequence[Any]) -> dict[str, Any]:
    return {name.id: convert_arg(arg) for name, arg in zip(params, args)}


def get_connectivity_args(
    neighbor_tables: Sequence[tuple[str, NeighborTableOffsetProvider]]
) -> dict[str, Any]:
    return {connectivity_identifier(offset): table.table for offset, table in neighbor_tables}


def get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    return {
        str(sym): size
        for name, value in args.items()
        for sym, size in zip(arrays[name].shape, value.shape)
    }


def get_offset_args(
    arrays: Mapping[str, dace.data.Array], params: Sequence[itir.Sym], args: Sequence[Any]
) -> Mapping[str, int]:
    return {
        str(sym): -drange.start
        for param, arg in zip(params, args)
        if is_field(arg)
        for sym, drange in zip(arrays[param.id].offset, get_sorted_dim_ranges(arg.domain))
    }


def get_stride_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    stride_args = {}
    for name, value in args.items():
        for sym, stride_size in zip(arrays[name].strides, value.strides):
            stride, remainder = divmod(stride_size, value.itemsize)
            if remainder != 0:
                raise ValueError(
                    f"Stride ({stride_size} bytes) for argument '{sym}' must be a multiple of item size ({value.itemsize} bytes)"
                )
            stride_args[str(sym)] = stride

    return stride_args


_build_cache_cpu: dict[str, CompiledSDFG] = {}
_build_cache_gpu: dict[str, CompiledSDFG] = {}


def get_cache_id(
    program: itir.FencilDefinition,
    arg_types: Sequence[ts.TypeSpec],
    column_axis: Optional[Dimension],
    offset_provider: Mapping[str, Any],
) -> str:
    max_neighbors = [
        (k, v.max_neighbors)
        for k, v in offset_provider.items()
        if isinstance(v, (NeighborTableOffsetProvider, StridedNeighborOffsetProvider))
    ]
    cache_id_args = [
        str(arg)
        for arg in (
            program,
            *arg_types,
            column_axis,
            *max_neighbors,
        )
    ]
    m = hashlib.sha256()
    for s in cache_id_args:
        m.update(s.encode())
    return m.hexdigest()


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    # build parameters
    auto_optimize = kwargs.get("auto_optimize", False)
    build_type = kwargs.get("build_type", "RelWithDebInfo")
    run_on_gpu = kwargs.get("run_on_gpu", False)
    build_cache = kwargs.get("build_cache", None)
    # ITIR parameters
    column_axis = kwargs.get("column_axis", None)
    offset_provider = kwargs["offset_provider"]

    arg_types = [type_translation.from_value(arg) for arg in args]
    neighbor_tables = filter_neighbor_tables(offset_provider)
    program_itir_not_working = [
        "calculate_horizontal_gradients_for_turbulence",
        "calculate_nabla2_and_smag_coefficients_for_vn",
        "calculate_nabla4",
        "mo_advection_traj_btraj_compute_o1_dsl",
        "mo_math_gradients_grad_green_gauss_cell_dsl",
        "mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1",
        "mo_solve_nonhydro_stencil_20",
        "mo_solve_nonhydro_stencil_21",
        "mo_solve_nonhydro_stencil_30",
        "mo_solve_nonhydro_stencil_41",
        "mo_velocity_advection_stencil_19",
        "temporary_fields_for_turbulence_diagnostics",
        "truly_horizontal_diffusion_nabla_of_theta_over_steep_points",
    ]

    validate_after_simplify = True
    cache_id = get_cache_id(program, arg_types, column_axis, offset_provider)
    if build_cache is not None and cache_id in build_cache:
        # retrieve SDFG program from build cache
        sdfg_program = build_cache[cache_id]
        sdfg = sdfg_program.sdfg
    else:
        # visit ITIR and generate SDFG
        if (
            any([ItirToSDFG._check_no_lifts(node) for node in program.closures])
            and program.id not in program_itir_not_working
        ):
            program_with_tmps: global_tmps.FencilWithTemporaries = preprocess_program(
                program, offset_provider, LiftMode.FORCE_TEMPORARIES
            )
            program = program_with_tmps.fencil
            tmps = program_with_tmps.tmps
            validate_after_simplify = False
        else:
            program = preprocess_program(program, offset_provider, LiftMode.FORCE_INLINE)
            tmps = []

        sdfg_genenerator = ItirToSDFG(arg_types, offset_provider, tmps, column_axis)
        sdfg = sdfg_genenerator.visit(program)
        sdfg.simplify(validate=validate_after_simplify)

        # set array storage for GPU execution
        if run_on_gpu:
            device = dace.DeviceType.GPU
            sdfg._name = f"{sdfg.name}_gpu"
            for _, _, array in sdfg.arrays_recursive():
                if not array.transient:
                    array.storage = dace.dtypes.StorageType.GPU_Global
        else:
            device = dace.DeviceType.CPU

        # run DaCe auto-optimization heuristics
        if auto_optimize:
            # TODO Investigate how symbol definitions improve autoopt transformations,
            #      in which case the cache table should take the symbols map into account.
            symbols: dict[str, int] = {}
            sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols)

        # compile SDFG and retrieve SDFG program
        sdfg.build_folder = cache._session_cache_dir_path / ".dacecache"
        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=build_type)
            dace.config.Config.set("compiler", "cpu", "args", value=_cpu_args)
            sdfg_program = sdfg.compile(validate=False)

        # store SDFG program in build cache
        if build_cache is not None:
            build_cache[cache_id] = sdfg_program

    dace_args = get_args(program.params, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_connectivity_args(neighbor_tables)
    dace_shapes = get_shape_args(sdfg.arrays, dace_field_args)
    dace_conn_shapes = get_shape_args(sdfg.arrays, dace_conn_args)
    dace_strides = get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_strides = get_stride_args(sdfg.arrays, dace_conn_args)
    dace_offsets = get_offset_args(sdfg.arrays, program.params, args)

    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_shapes,
        **dace_conn_shapes,
        **dace_strides,
        **dace_conn_strides,
        **dace_offsets,
    }
    expected_args = {
        key: value
        for key, value in all_args.items()
        if key in sdfg.signature_arglist(with_types=False)
    }

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg_program(**expected_args)


@program_executor
def run_dace_cpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(
        program,
        *args,
        **kwargs,
        build_cache=_build_cache_cpu,
        build_type=_build_type,
        run_on_gpu=False,
    )


@program_executor
def run_dace_gpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(
        program,
        *args,
        **kwargs,
        build_cache=_build_cache_gpu,
        build_type=_build_type,
        run_on_gpu=True,
    )
