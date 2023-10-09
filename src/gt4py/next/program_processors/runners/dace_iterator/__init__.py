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

from typing import Any, Dict, Mapping, Sequence, Tuple

import dace
import numpy as np
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.iterator.ir as itir
from gt4py.next import common
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.iterator.transforms import LiftMode, apply_common_transforms
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.type_system import type_translation

from .itir_to_sdfg import ItirToSDFG
from .utility import connectivity_identifier, filter_neighbor_tables


""" Default build configuration in DaCe backend """
_build_type = "Release"
# removing  -ffast-math from DaCe default compiler args in order to support isfinite/isinf/isnan built-ins
_cpu_args = (
    "-std=c++14 -fPIC -Wall -Wextra -O3 -march=native -Wno-unused-parameter -Wno-unused-label"
)


def convert_arg(arg: Any):
    if common.is_field(arg):
        sorted_dims = sorted(enumerate(arg.__gt_dims__), key=lambda v: v[1].value)
        ndim = len(sorted_dims)
        dim_indices = [dim[0] for dim in sorted_dims]
        assert isinstance(arg.ndarray, np.ndarray)
        return np.moveaxis(arg.ndarray, range(ndim), dim_indices)
    return arg


def preprocess_program(program: itir.FencilDefinition, offset_provider: Mapping[str, Any]):
    program = apply_common_transforms(
        program,
        offset_provider=offset_provider,
        lift_mode=LiftMode.FORCE_INLINE,
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


# TODO implement some hash of the FencilDefinition node as cache key, currently using the 'id' string
_build_cache: Dict[str, Tuple[dace.SDFG, CompiledSDFG]] = {}


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    # build parameters
    auto_optimize = kwargs.get("auto_optimize", False)
    build_type = kwargs.get("build_type", "RelWithDebInfo")
    run_on_gpu = kwargs.get("run_on_gpu", False)
    use_build_cache = kwargs.get("use_build_cache", False)
    # ITIR parameters
    column_axis = kwargs.get("column_axis", None)
    offset_provider = kwargs["offset_provider"]
    neighbor_tables = filter_neighbor_tables(offset_provider)

    if program.id in _build_cache:
        sdfg, sdfg_func = _build_cache[program.id]
    else:
        program = preprocess_program(program, offset_provider)
        arg_types = [type_translation.from_value(arg) for arg in args]
        sdfg_genenerator = ItirToSDFG(arg_types, offset_provider, column_axis)
        sdfg = sdfg_genenerator.visit(program)
        sdfg.simplify()

        if run_on_gpu:
            device = dace.DeviceType.GPU
            sdfg._name = f"{sdfg.name}_gpu"
            for _, _, array in sdfg.arrays_recursive():
                if not array.transient:
                    array.storage = dace.dtypes.StorageType.GPU_Global
        else:
            device = dace.DeviceType.CPU

        if auto_optimize:
            # TODO Investigate how symbol definitions improve autoopt transformations,
            #      in which case the cache table should take the symbols map into account.
            symbols: Dict[str, int] = {}
            sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols)

        sdfg.build_folder = cache._session_cache_dir_path / ".dacecache"
        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=build_type)
            dace.config.Config.set("compiler", "cpu", "args", value=_cpu_args)
            sdfg_func = sdfg.compile(validate=False)
        if use_build_cache:
            _build_cache[program.id] = (sdfg, sdfg_func)

    dace_args = get_args(program.params, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_connectivity_args(neighbor_tables)
    dace_shapes = get_shape_args(sdfg.arrays, dace_field_args)
    dace_conn_shapes = get_shape_args(sdfg.arrays, dace_conn_args)
    dace_strides = get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_strides = get_stride_args(sdfg.arrays, dace_conn_args)

    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_shapes,
        **dace_conn_shapes,
        **dace_strides,
        **dace_conn_strides,
    }
    expected_args = {
        key: value
        for key, value in all_args.items()
        if key in sdfg.signature_arglist(with_types=False)
    }

    if run_on_gpu:
        import cupy as cp

        sdfg._name = f"{sdfg.name}_gpu"
        for _, label, _ in sdfg.arrays_recursive():
            if not array.transient:
                host_array = dace_field_args[label]
                expected_args[label] = cp.array(host_array)

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg_func(**expected_args)

    if run_on_gpu:
        for k, v in expected_args.items():
            if k in dace_args and not cp.isscalar(v):
                np.copyto(dace_args[k], v.get())


@program_executor
def run_dace_cpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(program, *args, **kwargs, build_type=_build_type, run_on_gpu=False)


@program_executor
def run_dace_cpu_cached(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(
        program, *args, **kwargs, build_type=_build_type, run_on_gpu=False, use_build_cache=True
    )


@program_executor
def run_dace_gpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(program, *args, **kwargs, build_type=_build_type, run_on_gpu=True)


@program_executor
def run_dace_gpu_cached(program: itir.FencilDefinition, *args, **kwargs) -> None:
    run_dace_iterator(
        program, *args, **kwargs, build_type=_build_type, run_on_gpu=True, use_build_cache=True
    )
