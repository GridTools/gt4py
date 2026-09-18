# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Any, Final

from gt4py._core import definitions as core_defs, filecache
from gt4py.next import config, fingerprinting
from gt4py.next.otf import artifacts, recipes, stages, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace.workflow import bindings as bindings_step
from gt4py.next.program_processors.runners.dace.workflow.compilation import DaCeCompiler
from gt4py.next.program_processors.runners.dace.workflow.translation import DaCeTranslator


_GT_DACE_BINDING_FUNCTION_NAME: Final[str] = "update_sdfg_args"


def make_dace_translator(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    auto_optimize: bool = False,
    auto_optimize_args: dict[str, Any] | None = None,
    async_sdfg_call: bool = False,
    unstructured_horizontal_has_unit_stride: bool = False,
    use_metrics: bool = True,
    disable_field_origin_on_program_arguments: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> DaCeTranslator:
    """
    Build the GTIR -> SDFG translation step.

    Args:
        device_type: The device the compiled program targets.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        auto_optimize_args: Configuration for the auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU.
        unstructured_horizontal_has_unit_stride: Replace the field stride symbol
            with '1' in the horizontal dimension.
        use_metrics: Add SDFG instrumentation for stencil compute time.
        disable_field_origin_on_program_arguments: Assume zero-based field origins.
        use_max_domain_range_on_unstructured_shift: See `DaCeTranslator`.

    Returns:
        The configured translation step.
    """
    return DaCeTranslator(
        device_type=device_type,
        auto_optimize=auto_optimize,
        auto_optimize_args=auto_optimize_args,
        async_sdfg_call=async_sdfg_call,
        unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        use_metrics=use_metrics,
        disable_field_origin_on_program_arguments=disable_field_origin_on_program_arguments,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )


def make_dace_compile_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    auto_optimize: bool = False,
    cached_translation: bool = False,
    cmake_build_type: config.CMakeBuildType | None = None,
    translation: DaCeTranslator | None = None,
) -> recipes.OTFCompileWorkflow:
    """
    Build the DaCe translation -> bindings -> compilation workflow.

    Cross-cutting configuration is passed as keyword arguments and used to
    configure the steps this function creates. To customize the translation
    step, build one with `make_dace_translator` and pass it as `translation`;
    it is used verbatim, so it must agree with `device_type`.

    Args:
        device_type: The device the compiled program targets.
        auto_optimize: Enable the SDFG auto-optimize pipeline in the default
            translation step.
        cached_translation: Wrap the translation step in a persistent cache.
            Off by default; `make_dace_backend` turns it on.
        cmake_build_type: Build type for the generated project. Defaults to the
            value in `config`.
        translation: A pre-built translation step.

    Returns:
        The composed compile workflow.

    Raises:
        ValueError: If `translation` is configured for a different device.
    """
    if cmake_build_type is None:
        cmake_build_type = config.CMAKE_BUILD_TYPE

    if translation is None:
        bare_translation = make_dace_translator(
            device_type=device_type, auto_optimize=auto_optimize
        )
    else:
        workflow.check_device_agreement(translation, device_type, "DaCe translation step")
        bare_translation = translation

    translation_step: stages.TranslationStep
    if cached_translation:
        translation_step = workflow.CachedStep[
            stages.CompilableProgramDef, artifacts.ProgramSource, str
        ].persistent(
            bare_translation,
            input_fingerprinter=fingerprinting.strict_fingerprinter,
            cache=filecache.FileCache(
                cache.get_translation_cache_folder(
                    cache.get_cache_base_path(config.BUILD_CACHE_LIFETIME), "dace"
                )
            ),
        )
    else:
        translation_step = bare_translation

    return recipes.OTFCompileWorkflow(
        translation=translation_step,
        bindings=functools.partial(
            bindings_step.bind_sdfg, bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME
        ),
        compilation=DaCeCompiler(
            bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME,
            cache_lifetime=config.BUILD_CACHE_LIFETIME,
            device_type=device_type,
            cmake_build_type=cmake_build_type,
        ),
    )
