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

import functools
import warnings
from typing import Any, Callable, Optional

import factory

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
from gt4py.eve.utils import content_hash
from gt4py.next import NeighborTableOffsetProvider, backend, common, config
from gt4py.next.embedded.nd_array_field import NumPyArrayField
from gt4py.next.iterator import transforms
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors import modular_executor
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.type_system.type_translation import from_value


def handle_tuple(arg: Any, convert_arg: Callable) -> Any:
    return tuple(convert_arg(a) for a in arg)


def handle_field(arg: Any) -> tuple:
    arr = arg.ndarray
    origin = getattr(arg, "__gt_origin__", tuple([0] * len(arg.domain)))
    return arr, origin


type_handlers_convert_args = {
    tuple: handle_tuple,
    NumPyArrayField: handle_field,
}

try:
    import cupy as cp

    from gt4py.next.embedded.nd_array_field import CuPyArrayField

    type_handlers_convert_args[CuPyArrayField] = handle_field
except ImportError:
    cp = None


def handle_default(arg: Any) -> Any:
    return arg


def convert_arg(arg: Any) -> Any:
    handler = type_handlers_convert_args.get(type(arg), handle_default)  # type: ignore
    if handler is handle_tuple:
        return handler(arg, convert_arg)
    else:
        return handler(arg)


def convert_args(
    inp: stages.CompiledProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.CompiledProgram:
    def decorated_program(
        *args: Any,
        conn_args: Optional[list[ConnectivityArg]] = None,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
    ) -> None:
        # If we don't pass them as in the case of a CachedProgram extract connectivities here.
        if conn_args is None:
            conn_args = extract_connectivity_args(offset_provider, device)

        converted_args = [convert_arg(arg) for arg in args]
        return inp(*converted_args, *conn_args)

    return decorated_program


ConnectivityArg = tuple[core_defs.NDArrayObject, tuple[int, ...]]


def handle_connectivity(
    conn: NeighborTableOffsetProvider, zero_tuple: tuple[int, ...], device: core_defs.DeviceType
) -> ConnectivityArg:
    return (_ensure_is_on_device(conn.table, device), zero_tuple)


def handle_other_type(*args: Any, **kwargs: Any) -> None:
    return None


type_handlers_connectivity_args = {
    NeighborTableOffsetProvider: handle_connectivity,
    common.Dimension: handle_other_type,
}


def _ensure_is_on_device(
    connectivity_arg: core_defs.NDArrayObject, device: core_defs.DeviceType
) -> core_defs.NDArrayObject:
    if device == core_defs.DeviceType.CUDA:
        import cupy as cp

        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device.",
                stacklevel=2,
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def extract_connectivity_args(
    offset_provider: dict[str, Any], device: core_defs.DeviceType
) -> list[ConnectivityArg]:
    zero_tuple = (0, 0)
    args = []
    for conn in offset_provider.values():
        handler = type_handlers_connectivity_args.get(type(conn), handle_other_type)
        result = handler(conn, zero_tuple, device)  # type: ignore
        if result:
            args.append(result)
    return args


def compilation_hash(otf_closure: stages.ProgramCall) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = otf_closure.kwargs["offset_provider"]
    return hash(
        (
            otf_closure.program,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(from_value(arg) for arg in otf_closure.args)),
            id(offset_provider) if offset_provider else None,
            otf_closure.kwargs.get("column_axis", None),
        )
    )


class GTFNCompileWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(
            lambda: config.CMAKE_BUILD_TYPE
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

    translation = factory.SubFactory(
        gtfn_module.GTFNTranslationStepFactory, device_type=factory.SelfAttribute("..device_type")
    )
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableSource] = (
        nanobind.bind_source
    )
    compilation = factory.SubFactory(
        compiler.CompilerFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        builder_factory=factory.SelfAttribute("..builder_factory"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(convert_args, device=o.device_type)
    )


class GTFNBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_temps = ""
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: modular_executor.ModularExecutor(
                    otf_workflow=workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function),
                    name=o.name,
                )
            ),
            name_cached="_cached",
        )
        use_temporaries = factory.Trait(
            otf_workflow__translation__lift_mode=transforms.LiftMode.USE_TEMPORARIES,
            otf_workflow__translation__temporary_extraction_heuristics=global_tmps.SimpleTemporaryExtractionHeuristics,
            name_temps="_with_temporaries",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = compilation_hash
        otf_workflow = factory.SubFactory(
            GTFNCompileWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )
        name = factory.LazyAttribute(
            lambda o: f"run_gtfn_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
        )

    executor = factory.LazyAttribute(
        lambda o: modular_executor.ModularExecutor(otf_workflow=o.otf_workflow, name=o.name)
    )
    allocator = next_allocators.StandardCPUFieldBufferAllocator()


run_gtfn = GTFNBackendFactory()

run_gtfn_imperative = GTFNBackendFactory(
    name_postfix="_imperative", otf_workflow__translation__use_imperative_backend=True
)

run_gtfn_cached = GTFNBackendFactory(cached=True)

run_gtfn_with_temporaries = GTFNBackendFactory(use_temporaries=True)

run_gtfn_gpu = GTFNBackendFactory(gpu=True)

run_gtfn_gpu_cached = GTFNBackendFactory(gpu=True, cached=True)
