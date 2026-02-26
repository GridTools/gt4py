# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import factory
import numpy as np

import gt4py._core.definitions as core_defs
import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import filecache
from gt4py.next import backend, common, config, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


if TYPE_CHECKING:
    from gt4py.next import config_type


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
    inp: stages.ExecutableProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.ExecutableProgram:
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

        opt_kwargs: dict[str, Any] = {}
        if collect_metrics := metrics.is_level_enabled(metrics.PERFORMANCE):
            # If we are collecting metrics, we need to add the `exec_info` argument
            # to the `inp` call, which will be used to collect performance metrics.
            exec_info: dict[str, float] = {}
            opt_kwargs["exec_info"] = exec_info

        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        inp(
            *converted_args,
            *conn_args,
            **opt_kwargs,
        )

        if collect_metrics:
            metrics.add_sample_to_current_source(
                metrics.COMPUTE_METRIC, exec_info["run_cpp_duration"]
            )

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    zero_origin = (0, 0)
    assert all(
        hasattr(conn, "ndarray") or isinstance(conn, common.Dimension)
        for conn in offset_provider.values()
    )
    # Note: the order here needs to agree with the order of the generated bindings.
    # This is currently true only because when hashing offset provider dicts,
    # the keys' order is taken into account. Any modification to the hashing
    # of offset providers may break this assumption here.
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = [
        (ndarray, zero_origin)
        for conn in offset_provider.values()
        if (ndarray := getattr(conn, "ndarray", None)) is not None
    ]
    assert all(
        common.is_neighbor_table(conn) and field_utils.verify_device_field_type(conn, device)
        for conn in offset_provider.values()
        if hasattr(conn, "ndarray")
    )

    return args


class GTFNCompileWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config_type.CMakeBuildType = factory.LazyFunction(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda: config.cmake_build_type
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=stages.fingerprint_compilable_program,
                    cache=filecache.FileCache(str(config.build_cache_dir / "gtfn_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            gtfn_module.GTFNTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)

    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject] = (
        nanobind.bind_source  # type: ignore[has-type]  # mypy bug? mypy cannot see nanobind.bind_source type here
    )
    compilation = factory.SubFactory(
        compiler.CompilerFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.build_cache_lifetime),
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
