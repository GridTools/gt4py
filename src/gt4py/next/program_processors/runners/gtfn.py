# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any

import numpy as np

import gt4py._core.definitions as core_defs
import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import filecache
from gt4py.next import backend, common, config, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import definitions, recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import cache, compiler
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
    inp: stages.ExecutableProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.ExecutableProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.OffsetProviderElem],
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
    offset_provider: dict[str, common.OffsetProviderElem], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    zero_origin = (0, 0)
    assert all(hasattr(conn, "ndarray") for conn in offset_provider.values())
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


def make_gtfn_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    cached_translation: bool = False,
    enable_itir_transforms: bool = True,
    use_imperative_backend: bool = False,
    cmake_build_type: config.CMakeBuildType | None = None,
) -> recipes.OTFCompileWorkflow:
    """Build the GTFN translation -> bindings -> compilation -> decoration workflow."""
    cmake_build_type = config.CMAKE_BUILD_TYPE if cmake_build_type is None else cmake_build_type
    builder_factory: compiler.BuildSystemProjectGenerator = compiledb.CompiledbFactory(
        cmake_build_type=cmake_build_type
    )

    bare_translation = gtfn_module.GTFNTranslationStep(
        device_type=device_type,
        enable_itir_transforms=enable_itir_transforms,
        use_imperative_backend=use_imperative_backend,
    )
    translation: definitions.TranslationStep
    if cached_translation:
        translation = workflow.CachedStep.persistent(
            bare_translation,
            # mypy cannot solve `CachedStep`'s `HashT` type variable here (it only
            # appears in the fingerprinter's return), so the `str` fingerprint is
            # not recognized as a valid `HashT`.
            input_fingerprinter=stages.compilable_program_fingerprinter,  # type: ignore[arg-type]
            cache=filecache.FileCache(
                str(cache.get_cache_base_path(config.BUILD_CACHE_LIFETIME) / "gtfn_cache")
            ),
        )
    else:
        translation = bare_translation

    return recipes.OTFCompileWorkflow(
        translation=translation,
        bindings=nanobind.bind_source,
        compilation=compiler.Compiler(
            cache_lifetime=config.BUILD_CACHE_LIFETIME,
            builder_factory=builder_factory,
        ),
        decoration=functools.partial(convert_args, device=device_type),
    )


def make_gtfn_backend(
    *,
    gpu: bool = False,
    cached: bool = False,
    cached_translation: bool = False,
    enable_itir_transforms: bool = True,
    use_imperative_backend: bool = False,
    name_postfix: str = "",
    executor: workflow.Workflow[definitions.CompilableProgramDef, stages.ExecutableProgram]
    | None = None,
) -> backend.Backend:
    """Build a GTFN backend for the given device and caching configuration."""
    allocator: next_allocators.FieldBufferAllocatorProtocol
    device_type: core_defs.DeviceType
    if gpu:
        allocator = next_allocators.StandardGPUFieldBufferAllocator()
        device_type = core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA
        name_device = "gpu"
    else:
        allocator = next_allocators.StandardCPUFieldBufferAllocator()
        device_type = core_defs.DeviceType.CPU
        name_device = "cpu"

    otf_workflow = make_gtfn_workflow(
        device_type=device_type,
        cached_translation=cached_translation,
        enable_itir_transforms=enable_itir_transforms,
        use_imperative_backend=use_imperative_backend,
    )

    if executor is None:
        executor = (
            workflow.CachedStep.in_memory(
                otf_workflow, input_fingerprinter=stages.fast_compilable_program_fingerprinter
            )
            if cached
            else otf_workflow
        )

    name_cached = "_cached" if cached else ""
    return backend.Backend(
        name=f"run_gtfn_{name_device}{name_cached}{name_postfix}",
        executor=executor,
        allocator=allocator,
        transforms=backend.DEFAULT_TRANSFORMS,
    )


run_gtfn = make_gtfn_backend()

run_gtfn_imperative = make_gtfn_backend(name_postfix="_imperative", use_imperative_backend=True)

run_gtfn_cached = make_gtfn_backend(cached=True, cached_translation=True)

run_gtfn_gpu = make_gtfn_backend(gpu=True)

run_gtfn_gpu_cached = make_gtfn_backend(gpu=True, cached=True, cached_translation=True)

run_gtfn_no_transforms = make_gtfn_backend(enable_itir_transforms=False)
