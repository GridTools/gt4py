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
from gt4py.next import config
from gt4py.next.otf import definitions, recipes, stages, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace.workflow import (
    bindings as bindings_step,
    decoration as decoration_step,
)
from gt4py.next.program_processors.runners.dace.workflow.compilation import DaCeCompiler
from gt4py.next.program_processors.runners.dace.workflow.translation import DaCeTranslator


_GT_DACE_BINDING_FUNCTION_NAME: Final[str] = "update_sdfg_args"


def make_dace_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    auto_optimize: bool = False,
    cached_translation: bool = False,
    async_sdfg_call: bool = False,
    auto_optimize_args: dict[str, Any] | None = None,
    unstructured_horizontal_has_unit_stride: bool = False,
    use_metrics: bool = True,
    disable_field_origin_on_program_arguments: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
    cmake_build_type: config.CMakeBuildType | None = None,
) -> recipes.OTFCompileWorkflow:
    """Build the DaCe translation -> bindings -> compilation -> decoration workflow."""
    cmake_build_type = config.CMAKE_BUILD_TYPE if cmake_build_type is None else cmake_build_type

    bare_translation = DaCeTranslator(
        device_type=device_type,
        auto_optimize=auto_optimize,
        auto_optimize_args=auto_optimize_args,
        async_sdfg_call=async_sdfg_call,
        unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        use_metrics=use_metrics,
        disable_field_origin_on_program_arguments=disable_field_origin_on_program_arguments,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
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
                str(cache.get_cache_base_path(config.BUILD_CACHE_LIFETIME) / "translation_cache")
            ),
        )
    else:
        translation = bare_translation

    return recipes.OTFCompileWorkflow(
        translation=translation,
        bindings=functools.partial(
            bindings_step.bind_sdfg, bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME
        ),
        compilation=DaCeCompiler(
            bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME,
            cache_lifetime=config.BUILD_CACHE_LIFETIME,
            device_type=device_type,
            cmake_build_type=cmake_build_type,
        ),
        decoration=functools.partial(decoration_step.convert_args, device=device_type),
    )
