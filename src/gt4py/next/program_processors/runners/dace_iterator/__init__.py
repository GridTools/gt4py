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
import warnings
from inspect import currentframe, getframeinfo
from typing import Any, Mapping, Optional, Sequence

import dace
import numpy as np
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.allocators as next_allocators
import gt4py.next.iterator.ir as itir
import gt4py.next.program_processors.otf_compile_executor as otf_exec
import gt4py.next.program_processors.processor_interface as ppi
from gt4py.next import common
from gt4py.next.iterator import embedded as itir_embedded, transforms as itir_transforms
from gt4py.next.otf.compilation import cache as compilation_cache
from gt4py.next.type_system import type_specifications as ts, type_translation

from .itir_to_sdfg import ItirToSDFG
from .utility import connectivity_identifier, filter_neighbor_tables, get_sorted_dims


try:
    import cupy as cp
except ImportError:
    cp = None


def get_sorted_dim_ranges(domain: common.Domain) -> Sequence[common.FiniteUnitRange]:
    assert common.Domain.is_finite(domain)
    sorted_dims = get_sorted_dims(domain.dims)
    return [domain.ranges[dim_index] for dim_index, _ in sorted_dims]


""" Default build configuration in DaCe backend """
_build_type = "Release"


def convert_arg(arg: Any):
    if common.is_field(arg):
        sorted_dims = get_sorted_dims(arg.domain.dims)
        ndim = len(sorted_dims)
        dim_indices = [dim_index for dim_index, _ in sorted_dims]
        if isinstance(arg.ndarray, np.ndarray):
            return np.moveaxis(arg.ndarray, range(ndim), dim_indices)
        else:
            assert cp is not None and isinstance(arg.ndarray, cp.ndarray)
            return cp.moveaxis(arg.ndarray, range(ndim), dim_indices)
    return arg


def preprocess_program(
    program: itir.FencilDefinition,
    offset_provider: Mapping[str, Any],
    lift_mode: itir_transforms.LiftMode,
):
    node = itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        lift_mode=lift_mode,
        offset_provider=offset_provider,
        unroll_reduce=False,
    )
    # If we don't unroll, there may be lifts left in the itir which can't be lowered to SDFG.
    # In this case, just retry with unrolled reductions.
    if all([ItirToSDFG._check_no_lifts(closure) for closure in node.closures]):
        fencil_definition = node
    else:
        fencil_definition = itir_transforms.apply_common_transforms(
            program,
            common_subexpression_elimination=False,
            force_inline_lambda_args=True,
            lift_mode=lift_mode,
            offset_provider=offset_provider,
            unroll_reduce=True,
        )
    return fencil_definition


def get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    return {sdfg_param: convert_arg(arg) for sdfg_param, arg in zip(sdfg_params, args)}


def _ensure_is_on_device(
    connectivity_arg: np.typing.NDArray, device: dace.dtypes.DeviceType
) -> np.typing.NDArray:
    if device == dace.dtypes.DeviceType.GPU:
        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device."
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def get_connectivity_args(
    neighbor_tables: Sequence[tuple[str, itir_embedded.NeighborTableOffsetProvider]],
    device: dace.dtypes.DeviceType,
) -> dict[str, Any]:
    return {
        connectivity_identifier(offset): _ensure_is_on_device(table.table, device)
        for offset, table in neighbor_tables
    }


def get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    return {
        str(sym): size
        for name, value in args.items()
        for sym, size in zip(arrays[name].shape, value.shape)
    }


def get_offset_args(
    sdfg: dace.SDFG,
    args: Sequence[Any],
) -> Mapping[str, int]:
    sdfg_arrays: Mapping[str, dace.data.Array] = sdfg.arrays
    sdfg_params: Sequence[str] = sdfg.arg_names
    return {
        str(sym): -drange.start
        for sdfg_param, arg in zip(sdfg_params, args)
        if common.is_field(arg)
        for sym, drange in zip(sdfg_arrays[sdfg_param].offset, get_sorted_dim_ranges(arg.domain))
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
                    f"Stride ({stride_size} bytes) for argument '{sym}' must be a multiple of item size ({value.itemsize} bytes)."
                )
            stride_args[str(sym)] = stride

    return stride_args


_build_cache_cpu: dict[str, CompiledSDFG] = {}
_build_cache_gpu: dict[str, CompiledSDFG] = {}


def get_cache_id(
    program: itir.FencilDefinition,
    arg_types: Sequence[ts.TypeSpec],
    column_axis: Optional[common.Dimension],
    offset_provider: Mapping[str, Any],
) -> str:
    max_neighbors = [
        (k, v.max_neighbors)
        for k, v in offset_provider.items()
        if isinstance(
            v,
            (
                itir_embedded.NeighborTableOffsetProvider,
                itir_embedded.StridedNeighborOffsetProvider,
            ),
        )
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


def get_sdfg_args(sdfg: dace.SDFG, *args, **kwargs) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the same arguments that are passed to `run_dace_iterator()`.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
    """  # noqa: D401
    offset_provider = kwargs["offset_provider"]
    on_gpu = kwargs.get("on_gpu", False)

    neighbor_tables = filter_neighbor_tables(offset_provider)
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    sdfg_sig = sdfg.signature_arglist(with_types=False)
    dace_args = get_args(sdfg, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_connectivity_args(neighbor_tables, device)
    dace_shapes = get_shape_args(sdfg.arrays, dace_field_args)
    dace_conn_shapes = get_shape_args(sdfg.arrays, dace_conn_args)
    dace_strides = get_stride_args(sdfg.arrays, dace_field_args)
    dace_conn_strides = get_stride_args(sdfg.arrays, dace_conn_args)
    dace_offsets = get_offset_args(sdfg, args)
    all_args = {
        **dace_args,
        **dace_conn_args,
        **dace_shapes,
        **dace_conn_shapes,
        **dace_strides,
        **dace_conn_strides,
        **dace_offsets,
    }
    expected_args = {key: all_args[key] for key in sdfg_sig}

    return expected_args


def build_sdfg_from_itir(
    program: itir.FencilDefinition,
    *args,
    offset_provider: dict[str, Any],
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: itir_transforms.LiftMode = itir_transforms.LiftMode.FORCE_INLINE,
) -> dace.SDFG:
    """Translate a Fencil into an SDFG.

    Args:
        program:	        The Fencil that should be translated.
        *args:		        Arguments for which the fencil should be called.
        offset_provider:	The set of offset providers that should be used.
        auto_optimize:	    Apply DaCe's `auto_optimize` heuristic.
        on_gpu:		        Performs the translation for GPU, defaults to `False`.
        column_axis:		The column axis to be used, defaults to `None`.
        lift_mode:		    Which lift mode should be used, defaults `FORCE_INLINE`.

    Notes:
        Currently only the `FORCE_INLINE` liftmode is supported and the value of `lift_mode` is ignored.
    """
    # TODO(edopao): As temporary fix until temporaries are supported in the DaCe Backend force
    #                `lift_more` to `FORCE_INLINE` mode.
    lift_mode = itir_transforms.LiftMode.FORCE_INLINE
    arg_types = [type_translation.from_value(arg) for arg in args]

    # visit ITIR and generate SDFG
    program = preprocess_program(program, offset_provider, lift_mode)
    sdfg_genenerator = ItirToSDFG(arg_types, offset_provider, column_axis)
    sdfg = sdfg_genenerator.visit(program)
    if sdfg is None:
        raise RuntimeError(f"Visit failed for program {program.id}.")

    for nested_sdfg in sdfg.all_sdfgs_recursive():
        if not nested_sdfg.debuginfo:
            _, frameinfo = warnings.warn(
                f"{nested_sdfg} does not have debuginfo. Consider adding them in the corresponding nested sdfg."
            ), getframeinfo(
                currentframe()  # type: ignore
            )
            nested_sdfg.debuginfo = dace.dtypes.DebugInfo(
                start_line=frameinfo.lineno,
                end_line=frameinfo.lineno,
                filename=frameinfo.filename,
            )

    # run DaCe transformations to simplify the SDFG
    sdfg.simplify()

    # run DaCe auto-optimization heuristics
    if auto_optimize:
        # TODO: Investigate how symbol definitions improve autoopt transformations,
        #       in which case the cache table should take the symbols map into account.
        symbols: dict[str, int] = {}
        device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU
        sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols, use_gpu_storage=on_gpu)

    if on_gpu:
        sdfg.apply_gpu_transformations()

    return sdfg


def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs):
    # build parameters
    build_cache = kwargs.get("build_cache", None)
    compiler_args = kwargs.get("compiler_args", None)  # `None` will take default.
    build_type = kwargs.get("build_type", "RelWithDebInfo")
    on_gpu = kwargs.get("on_gpu", False)
    auto_optimize = kwargs.get("auto_optimize", True)
    lift_mode = kwargs.get("lift_mode", itir_transforms.LiftMode.FORCE_INLINE)
    # ITIR parameters
    column_axis = kwargs.get("column_axis", None)
    offset_provider = kwargs["offset_provider"]

    arg_types = [type_translation.from_value(arg) for arg in args]

    cache_id = get_cache_id(program, arg_types, column_axis, offset_provider)
    if build_cache is not None and cache_id in build_cache:
        # retrieve SDFG program from build cache
        sdfg_program = build_cache[cache_id]
        sdfg = sdfg_program.sdfg

    else:
        sdfg = build_sdfg_from_itir(
            program,
            *args,
            offset_provider=offset_provider,
            auto_optimize=auto_optimize,
            on_gpu=on_gpu,
            column_axis=column_axis,
            lift_mode=lift_mode,
        )

        sdfg.build_folder = compilation_cache._session_cache_dir_path / ".dacecache"
        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=build_type)
            if compiler_args is not None:
                dace.config.Config.set(
                    "compiler", "cuda" if on_gpu else "cpu", "args", value=compiler_args
                )
            sdfg_program = sdfg.compile(validate=False)

        # store SDFG program in build cache
        if build_cache is not None:
            build_cache[cache_id] = sdfg_program

    expected_args = get_sdfg_args(sdfg, *args, **kwargs)

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg_program(**expected_args)


def _run_dace_cpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
    compiler_args = dace.config.Config.get("compiler", "cpu", "args")

    # disable finite-math-only in order to support isfinite/isinf/isnan builtins
    if "-ffast-math" in compiler_args:
        compiler_args += " -fno-finite-math-only"
    if "-ffinite-math-only" in compiler_args:
        compiler_args.replace("-ffinite-math-only", "")

    run_dace_iterator(
        program,
        *args,
        **kwargs,
        build_cache=_build_cache_cpu,
        build_type=_build_type,
        compiler_args=compiler_args,
        on_gpu=False,
    )


run_dace_cpu = otf_exec.OTFBackend(
    executor=ppi.program_executor(_run_dace_cpu, name="run_dace_cpu"),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
)

if cp:

    def _run_dace_gpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
        run_dace_iterator(
            program,
            *args,
            **kwargs,
            build_cache=_build_cache_gpu,
            build_type=_build_type,
            on_gpu=True,
        )

else:

    def _run_dace_gpu(program: itir.FencilDefinition, *args, **kwargs) -> None:
        raise RuntimeError("Missing 'cupy' dependency for GPU execution.")


run_dace_gpu = otf_exec.OTFBackend(
    executor=ppi.program_executor(_run_dace_gpu, name="run_dace_gpu"),
    allocator=next_allocators.StandardGPUFieldBufferAllocator(),
)
