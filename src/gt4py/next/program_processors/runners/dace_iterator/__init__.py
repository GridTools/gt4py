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
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import dace
import numpy as np
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.sdfg import utils as sdutils
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.allocators as next_allocators
import gt4py.next.iterator.ir as itir
import gt4py.next.program_processors.otf_compile_executor as otf_exec
import gt4py.next.program_processors.processor_interface as ppi
from gt4py.next import common
from gt4py.next.iterator import transforms as itir_transforms
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


def convert_arg(arg: Any, sdfg_param: str):
    if common.is_field(arg):
        # field domain offsets are not supported
        non_zero_offsets = [
            (dim, dim_range)
            for dim, dim_range in zip(arg.domain.dims, arg.domain.ranges)
            if dim_range.start != 0
        ]
        if non_zero_offsets:
            dim, dim_range = non_zero_offsets[0]
            raise RuntimeError(
                f"Field '{sdfg_param}' passed as array slice with offset {dim_range.start} on dimension {dim.value}."
            )
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
    unroll_reduce: bool = False,
):
    node = itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        force_inline_lambda_args=True,
        lift_mode=lift_mode,
        offset_provider=offset_provider,
        unroll_reduce=unroll_reduce,
    )

    if isinstance(node, itir_transforms.global_tmps.FencilWithTemporaries):
        fencil_definition = node.fencil
        tmps = node.tmps

    elif isinstance(node, itir.FencilDefinition):
        fencil_definition = node
        tmps = []

    else:
        raise TypeError(
            f"Expected 'FencilDefinition' or 'FencilWithTemporaries', got '{type(program).__name__}'."
        )

    return fencil_definition, tmps


def get_args(sdfg: dace.SDFG, args: Sequence[Any]) -> dict[str, Any]:
    sdfg_params: Sequence[str] = sdfg.arg_names
    return {sdfg_param: convert_arg(arg, sdfg_param) for sdfg_param, arg in zip(sdfg_params, args)}


def _ensure_is_on_device(
    connectivity_arg: np.typing.NDArray, device: dace.dtypes.DeviceType
) -> np.typing.NDArray:
    if device == dace.dtypes.DeviceType.GPU:
        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device.",
                stacklevel=2,
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def get_connectivity_args(
    neighbor_tables: Mapping[str, common.NeighborTable],
    device: dace.dtypes.DeviceType,
) -> dict[str, Any]:
    return {
        connectivity_identifier(offset): _ensure_is_on_device(offset_provider.table, device)
        for offset, offset_provider in neighbor_tables.items()
    }


def get_shape_args(
    arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]
) -> Mapping[str, int]:
    shape_args: dict[str, int] = {}
    for name, value in args.items():
        for sym, size in zip(arrays[name].shape, value.shape):
            if isinstance(sym, dace.symbol):
                assert sym.name not in shape_args
                shape_args[sym.name] = size
            elif sym != size:
                raise RuntimeError(
                    f"Expected shape {arrays[name].shape} for arg {name}, got {value.shape}."
                )
    return shape_args


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


_build_cache: dict[str, CompiledSDFG] = {}


def get_cache_id(
    build_type: str,
    build_for_gpu: bool,
    lift_mode: itir_transforms.LiftMode,
    program: itir.FencilDefinition,
    arg_types: Sequence[ts.TypeSpec],
    column_axis: Optional[common.Dimension],
    offset_provider: Mapping[str, Any],
) -> str:
    def offset_invariants(offset):
        if isinstance(offset, common.Connectivity):
            return (
                offset.origin_axis,
                offset.neighbor_axis,
                offset.has_skip_values,
                offset.max_neighbors,
            )
        if isinstance(offset, common.Dimension):
            return (offset,)
        return tuple()

    offset_cache_keys = [
        (name, *offset_invariants(offset)) for name, offset in offset_provider.items()
    ]
    cache_id_args = [
        str(arg)
        for arg in (
            build_type,
            build_for_gpu,
            lift_mode,
            program,
            *arg_types,
            column_axis,
            *offset_cache_keys,
        )
    ]
    m = hashlib.sha256()
    for s in cache_id_args:
        m.update(s.encode())
    return m.hexdigest()


def get_sdfg_args(sdfg: dace.SDFG, *args, check_args: bool = False, **kwargs) -> dict[str, Any]:
    """Extracts the arguments needed to call the SDFG.

    This function can handle the same arguments that are passed to `run_dace_iterator()`.

    Args:
        sdfg:               The SDFG for which we want to get the arguments.
    """
    offset_provider = kwargs["offset_provider"]
    on_gpu = kwargs.get("on_gpu", False)

    neighbor_tables = filter_neighbor_tables(offset_provider)
    device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU

    dace_args = get_args(sdfg, args)
    dace_field_args = {n: v for n, v in dace_args.items() if not np.isscalar(v)}
    dace_conn_args = get_connectivity_args(neighbor_tables, device)
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

    if check_args:
        # return only arguments expected in SDFG signature (note hat `signature_arglist` takes time)
        sdfg_sig = sdfg.signature_arglist(with_types=False)
        return {key: all_args[key] for key in sdfg_sig}

    return all_args


def build_sdfg_from_itir(
    program: itir.FencilDefinition,
    *args,
    offset_provider: dict[str, Any],
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: itir_transforms.LiftMode = itir_transforms.LiftMode.FORCE_INLINE,
    load_sdfg_from_file: bool = False,
    cache_id: Optional[str] = None,
    save_sdfg: bool = True,
) -> dace.SDFG:
    """Translate a Fencil into an SDFG.

    Args:
        program:             The Fencil that should be translated.
        *args:               Arguments for which the fencil should be called.
        offset_provider:     The set of offset providers that should be used.
        auto_optimize:       Apply DaCe's `auto_optimize` heuristic.
        on_gpu:              Performs the translation for GPU, defaults to `False`.
        column_axis:         The column axis to be used, defaults to `None`.
        lift_mode:           Which lift mode should be used, defaults `FORCE_INLINE`.
        load_sdfg_from_file: Allows to read the SDFG from file, instead of generating it, for debug only.
        cache_id:            The id of the cache entry, used to disambiguate stored sdfgs.
        save_sdfg:           If `True`, the default the SDFG is stored as a file and can be loaded, this allows to skip the lowering step, requires `load_sdfg_from_file` set to `True`.

    Notes:
        Currently only the `FORCE_INLINE` liftmode is supported and the value of `lift_mode` is ignored.
    """
    # Test if we can go through the cache?
    sdfg_filename = (
        f"_dacegraphs/gt4py/{cache_id if cache_id is not None else '.'}/{program.id}.sdfg"
    )
    if load_sdfg_from_file and Path(sdfg_filename).exists():
        sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_filename)
        sdfg.validate()
        return sdfg

    arg_types = [type_translation.from_value(arg) for arg in args]

    # visit ITIR and generate SDFG
    program, tmps = preprocess_program(program, offset_provider, lift_mode)
    sdfg_genenerator = ItirToSDFG(arg_types, offset_provider, tmps, column_axis)
    sdfg = sdfg_genenerator.visit(program)
    if sdfg is None:
        raise RuntimeError(f"Visit failed for program {program.id}.")

    for nested_sdfg in sdfg.all_sdfgs_recursive():
        if not nested_sdfg.debuginfo:
            _, frameinfo = (
                warnings.warn(
                    f"{nested_sdfg.label} does not have debuginfo. Consider adding them in the corresponding nested sdfg.",
                    stacklevel=2,
                ),
                getframeinfo(currentframe()),  # type: ignore[arg-type]
            )
            nested_sdfg.debuginfo = dace.dtypes.DebugInfo(
                start_line=frameinfo.lineno,
                end_line=frameinfo.lineno,
                filename=frameinfo.filename,
            )

    # TODO(edopao): remove `inline_loop_blocks` when DaCe transformations support LoopRegion construct
    sdutils.inline_loop_blocks(sdfg)

    # run DaCe transformations to simplify the SDFG
    sdfg.simplify()

    # run DaCe auto-optimization heuristics
    if auto_optimize:
        # TODO: Investigate performance improvement from SDFG specialization with constant symbols,
        #       for array shape and strides, although this would imply JIT compilation.
        symbols: dict[str, int] = {}
        device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU
        sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols, use_gpu_storage=on_gpu)
    elif on_gpu:
        autoopt.apply_gpu_storage(sdfg)

    if on_gpu:
        sdfg.apply_gpu_transformations()

    # Store the sdfg such that we can later reuse it.
    if save_sdfg:
        sdfg.save(sdfg_filename)

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
    # debug option to store SDFGs on filesystem and skip lowering ITIR to SDFG at each run
    load_sdfg_from_file = kwargs.get("load_sdfg_from_file", False)
    save_sdfg = kwargs.get("save_sdfg", True)

    arg_types = [type_translation.from_value(arg) for arg in args]

    cache_id = get_cache_id(
        build_type, on_gpu, lift_mode, program, arg_types, column_axis, offset_provider
    )
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
            load_sdfg_from_file=load_sdfg_from_file,
            cache_id=cache_id,
            save_sdfg=save_sdfg,
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

    sdfg_args = get_sdfg_args(sdfg, *args, **kwargs)

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg_program(**sdfg_args)


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
        build_cache=_build_cache,
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
            build_cache=_build_cache,
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
