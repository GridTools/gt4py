# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Any, Final, TypedDict

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import recipes, stages
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace.workflow import bindings as bindings_step
from gt4py.next.program_processors.runners.dace.workflow.compilation import DaCeCompiler
from gt4py.next.program_processors.runners.dace.workflow.translation import DaCeTranslator


_GT_DACE_BINDING_FUNCTION_NAME: Final[str] = "update_sdfg_args"


class DaCeTranslationOptions(TypedDict, total=False):
    """
    Step-local settings of `DaCeTranslator`.

    The device, auto-optimize and unit-stride settings come from the builder.
    """

    auto_optimize_args: dict[str, Any] | None
    async_sdfg_call: bool
    use_metrics: bool
    disable_itir_transforms: bool
    disable_field_origin_on_program_arguments: bool
    use_max_domain_range_on_unstructured_shift: bool | None


class DaCeCompilationOptions(TypedDict, total=False):
    """Step-local settings of `DaCeCompiler`; device, build type and cache lifetime come from the builder."""

    add_gpu_trace_markers: bool


#: Defaults of the `DaCeTranslator` fields that have no dataclass default.
_DEFAULT_TRANSLATION_OPTIONS: Final[DaCeTranslationOptions] = DaCeTranslationOptions(
    auto_optimize_args=None, async_sdfg_call=True, use_metrics=True
)


def make_dace_compile_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    auto_optimize: bool = False,
    cached_translation: bool = False,
    cmake_build_type: config.CMakeBuildType | None = None,
    unstructured_horizontal_has_unit_stride: bool | None = None,
    translation: DaCeTranslationOptions | None = None,
    compilation: DaCeCompilationOptions | None = None,
) -> recipes.OTFCompileWorkflow:
    """
    Build the DaCe translation -> bindings -> compilation workflow.

    Settings shared by several steps are keyword arguments, forwarded to every
    step that needs them. The step-local settings of each step are passed as a
    dict, unpacked into the step's constructor; the shared settings are not
    part of these dicts, so they cannot be set for one step alone. To replace a
    whole step, use `dataclasses.replace` on the returned workflow.

    Args:
        device_type: The device the compiled program targets.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        cached_translation: Wrap the translation step in a persistent cache.
        cmake_build_type: Build type of the generated project. Defaults to the
            value in `config`.
        unstructured_horizontal_has_unit_stride: Replace the field stride
            symbol with '1' in the horizontal dimension. Defaults to the value
            in `config`.
        translation: Step-local settings of the translation step.
        compilation: Step-local settings of the compilation step.

    Returns:
        The composed compile workflow.
    """
    if cmake_build_type is None:
        cmake_build_type = config.CMAKE_BUILD_TYPE
    if unstructured_horizontal_has_unit_stride is None:
        unstructured_horizontal_has_unit_stride = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE

    translation_step: stages.TranslationStep = DaCeTranslator(
        device_type=device_type,
        auto_optimize=auto_optimize,
        unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        **(_DEFAULT_TRANSLATION_OPTIONS | (translation or DaCeTranslationOptions())),
    )
    if cached_translation:
        translation_step = cache.persistent_translation_cache(translation_step, "dace")

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
            **(compilation or DaCeCompilationOptions()),
        ),
    )
