# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import pathlib
import warnings
from collections.abc import Callable
from typing import Any, ClassVar, Final, TypeAlias

import gt4py
from gt4py.next import backend as next_backend, config
from gt4py.next.otf import artifacts, recipes, stages, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.workflow import (
    bindings as bindings_step,
    common as gtx_wfdcommon,
)
from gt4py.next.program_processors.runners.dace.workflow.compilation import DaCeCompiler
from gt4py.next.program_processors.runners.dace.workflow.translation import DaCeTranslator


#: Warnings about the builder arguments point at the first caller outside GT4Py.
_GT4PY_SOURCE_PREFIX: Final[str] = str(pathlib.Path(gt4py.__file__).parent)


@dataclasses.dataclass(frozen=True)
class DaCeConfig(next_backend.ToolchainConfig):
    """Settings shared by the steps of a DaCe toolchain, see `ToolchainConfig`."""

    #: Run the SDFG auto-optimize pipeline.
    auto_optimize: bool = True
    #: Workspace memory allocated outside the SDFG. The toolchain injects it into
    #: the loaded programs, and the translation step then defaults to the
    #: `EXTERNAL` transient memory mode, which stores the transients in it. The
    #: workspace is a dict, so a config holding one is not hashable.
    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None

    #: Name of the function that binds the SDFG arguments. The bindings and the
    #: compilation steps must agree on it.
    bind_func_name: ClassVar[str] = "update_sdfg_args"


def make_dace_translator(
    cfg: DaCeConfig,
    /,
    *,
    auto_optimize_args: dict[str, Any] | None = None,
    async_sdfg_call: bool = True,
    use_metrics: bool = True,
    disable_itir_transforms: bool = False,
    disable_field_origin_on_program_arguments: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> DaCeTranslator:
    """
    Build the GTIR -> SDFG translation step.

    The keyword arguments are the step-local fields of `DaCeTranslator`.

    Args:
        cfg: The toolchain configuration.
        auto_optimize_args: Configuration for the SDFG auto-optimize pipeline,
            see `gt_auto_optimize()`. The parameters `DaCeTranslator` derives
            itself cannot be set here. With an external workspace in `cfg`, the
            `transient_memory_mode` defaults to `EXTERNAL`.
        async_sdfg_call: Make an asynchronous SDFG call, to overlap GPU kernel
            execution with the Python driver code. Only effective on GPU.
        use_metrics: Add SDFG instrumentation for stencil compute time.
        disable_itir_transforms: Skip the GTIR transformation passes.
        disable_field_origin_on_program_arguments: Assume that all fields passed
            as program arguments have zero-based origin, which skips the range
            start-symbols `_range_0`.
        use_max_domain_range_on_unstructured_shift: See `DaCeTranslator`.

    Returns:
        The translation step, targeting `cfg.device_type`.

    Raises:
        ValueError: If `auto_optimize_args` sets a parameter `DaCeTranslator`
            derives itself, or requests the `EXTERNAL` transient memory mode
            without an external workspace in `cfg`.
    """
    optimization_args = dict(auto_optimize_args or {})
    if optimization_args and not cfg.auto_optimize:
        warnings.warn(
            "Optimizations args given, but auto-optimize is disabled.",
            skip_file_prefixes=(_GT4PY_SOURCE_PREFIX,),
        )

    if cfg.external_workspace is None:
        if (
            optimization_args.get("transient_memory_mode")
            is gtx_transformations.TransientMemoryMode.EXTERNAL
        ):
            raise ValueError(
                "External memory workspace must be provided when 'transient_memory_mode' is 'EXTERNAL'."
            )
    elif transient_memory_mode := optimization_args.get("transient_memory_mode"):
        if transient_memory_mode is not gtx_transformations.TransientMemoryMode.EXTERNAL:
            warnings.warn(
                f"External memory workspace provided but 'transient_memory_mode' is '{transient_memory_mode}', it requires '{gtx_transformations.TransientMemoryMode.EXTERNAL}'.",
                skip_file_prefixes=(_GT4PY_SOURCE_PREFIX,),
            )
    else:
        optimization_args["transient_memory_mode"] = (
            gtx_transformations.TransientMemoryMode.EXTERNAL
        )

    return DaCeTranslator(
        device_type=cfg.device_type,
        auto_optimize=cfg.auto_optimize,
        auto_optimize_args=optimization_args,
        async_sdfg_call=async_sdfg_call and cfg.gpu,
        unstructured_horizontal_has_unit_stride=cfg.unstructured_horizontal_has_unit_stride,
        use_metrics=use_metrics,
        disable_itir_transforms=disable_itir_transforms,
        disable_field_origin_on_program_arguments=disable_field_origin_on_program_arguments,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )


def make_dace_bindings(
    cfg: DaCeConfig, /
) -> workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]:
    """Build the step generating the bindings of the translated SDFG."""
    return functools.partial(bindings_step.bind_sdfg, bind_func_name=cfg.bind_func_name)


def make_dace_compiler(
    cfg: DaCeConfig, /, *, add_gpu_trace_markers: bool | None = None
) -> DaCeCompiler:
    """
    Build the compilation step, targeting `cfg.device_type`.

    Args:
        cfg: The toolchain configuration.
        add_gpu_trace_markers: Add GPU trace markers to the generated code.
            Defaults to the value in `config`.

    Returns:
        The compilation step.
    """
    if add_gpu_trace_markers is None:
        add_gpu_trace_markers = config.ADD_GPU_TRACE_MARKERS
    return DaCeCompiler(
        bind_func_name=cfg.bind_func_name,
        cache_lifetime=cfg.cache_lifetime,
        device_type=cfg.device_type,
        cmake_build_type=cfg.cmake_build_type,
        add_gpu_trace_markers=add_gpu_trace_markers,
    )


#: Step builders: callables creating a step from the toolchain configuration.
DaCeTranslationBuilder: TypeAlias = Callable[[DaCeConfig], stages.TranslationStep]
DaCeBindingsBuilder: TypeAlias = Callable[
    [DaCeConfig], workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]
]
DaCeCompilationBuilder: TypeAlias = Callable[
    [DaCeConfig], workflow.Workflow[artifacts.ExtensionSource, artifacts.CompilationArtifact]
]


def make_dace_compile_workflow(
    cfg: DaCeConfig | None = None,
    /,
    *,
    translation: DaCeTranslationBuilder = make_dace_translator,
    bindings: DaCeBindingsBuilder = make_dace_bindings,
    compilation: DaCeCompilationBuilder = make_dace_compiler,
) -> recipes.OTFCompileWorkflow:
    """
    Build the DaCe translation -> bindings -> compilation workflow.

    Every step is created by a step builder that receives `cfg`, so all steps
    agree on the settings in it. To customize a step, pass a different
    builder: a `functools.partial` of the default one to change a step-local
    setting, e.g. `translation=functools.partial(make_dace_translator, use_metrics=False)`,
    or any callable taking the config to replace the step. The translation
    step is wrapped in the cache here, after its builder ran, so a custom
    translation step is cached like the default one. A custom step builder is
    responsible for configuring its step from `cfg`.

    Args:
        cfg: The toolchain configuration. Defaults to `DaCeConfig()`.
        translation: Builder of the translation step.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The composed compile workflow.
    """
    if cfg is None:
        cfg = DaCeConfig()

    translation_step = translation(cfg)
    compilation_step = compilation(cfg)

    if cfg.cached_translation:
        translation_step = cache.persistent_translation_cache(
            translation_step, "dace", cfg.cache_lifetime
        )

    return recipes.OTFCompileWorkflow(
        translation=translation_step, bindings=bindings(cfg), compilation=compilation_step
    )
