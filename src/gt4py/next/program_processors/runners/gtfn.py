# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any

import factory
import numpy as np

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
from gt4py._core import filecache
from gt4py.next import backend, common, config, field_utils, metrics
from gt4py.next.embedded import nd_array_field
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


def convert_arg(arg: Any) -> Any:
    # Note: this function is on the hot path and needs to have minimal overhead.
    if (origin := getattr(arg, "__gt_origin__", None)) is not None:
        # `Field` is the most likely case, we use `__gt_origin__` as the property is needed anyway
        # and (currently) uniquely identifies a `NDArrayField` (which is the only supported `Field`)
        assert isinstance(arg, nd_array_field.NdArrayField)
        return arg.ndarray, origin
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, np.bool_):
        # nanobind does not support implicit conversion of `np.bool` to `bool`
        return bool(arg)
    # TODO(havogt): if this function still appears in profiles,
    # we should avoid going through the previous isinstance checks for detecting a scalar.
    # E.g. functools.cache on the arg type, returning a function that does the conversion
    return arg


def convert_args(
    inp: stages.CompiledProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.CompiledProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
        out: Any = None,
    ) -> None:
        # Note: this function is on the hot path and needs to have minimal overhead.
        if out is not None:
            args = (*args, out)
        converted_args = (convert_arg(arg) for arg in args)
        conn_args = extract_connectivity_args(offset_provider, device)

        opt_kwargs: dict[str, Any]
        metric_collection = metrics.get_active_metric_collection()
        if collect_metrics := (
            metric_collection is not None and (config.COLLECT_METRICS_LEVEL >= metrics.PERFORMANCE)
        ):
            # If we are collecting metrics, we need to add the `exec_info` argument
            # to the `inp` call, which will be used to collect performance metrics.
            exec_info: dict[str, float] = {}
            opt_kwargs = {"exec_info": exec_info}
        else:
            opt_kwargs = {}

        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        inp(
            *converted_args,
            *conn_args,
            **opt_kwargs,
        )

        if collect_metrics:
            assert metric_collection is not None
            value = exec_info["run_cpp_end_time"] - exec_info["run_cpp_start_time"]
            metric_collection.add_sample(metrics.COMPUTE_METRIC, value)

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = []
    # Note: the order here needs to agree with the order of the generated bindings
    for conn in offset_provider.values():
        if (ndarray := getattr(conn, "ndarray", None)) is not None:
            assert common.is_neighbor_table(conn)
            assert field_utils.verify_device_field_type(conn, device)
            args.append((ndarray, (0, 0)))
            continue
        assert isinstance(conn, common.Dimension)
    return args


class GTFNCompileWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda: config.CMAKE_BUILD_TYPE
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=stages.fingerprint_compilable_program,
                    cache=filecache.FileCache(str(config.BUILD_CACHE_DIR / "gtfn_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            gtfn_module.GTFNTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)

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
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = stages.compilation_hash
        otf_workflow = factory.SubFactory(
            GTFNCompileWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )

    name = factory.LazyAttribute(
        lambda o: f"run_gtfn_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


run_gtfn = GTFNBackendFactory()

run_gtfn_imperative = GTFNBackendFactory(
    name_postfix="_imperative", otf_workflow__translation__use_imperative_backend=True
)

run_gtfn_cached = GTFNBackendFactory(cached=True, otf_workflow__cached_translation=True)

run_gtfn_gpu = GTFNBackendFactory(gpu=True)

run_gtfn_gpu_cached = GTFNBackendFactory(
    gpu=True, cached=True, otf_workflow__cached_translation=True
)

run_gtfn_no_transforms = GTFNBackendFactory(
    otf_workflow__bare_translation__enable_itir_transforms=False
)
