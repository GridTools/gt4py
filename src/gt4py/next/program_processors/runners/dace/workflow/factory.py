# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Final

from gt4py._core import definitions as core_defs, filecache
from gt4py.next import config
from gt4py.next.otf import definitions, recipes, stages, workflow
from gt4py.next.otf.compilation import cache
from gt4py.next.program_processors.runners.dace.workflow import (
    bindings as bindings_step,
    decoration as decoration_step,
)
from gt4py.next.program_processors.runners.dace.workflow.compilation import DaCeCompiler
from gt4py.next.program_processors.runners.dace.workflow.translation import make_dace_translator


_GT_DACE_BINDING_FUNCTION_NAME: Final[str] = "update_sdfg_args"


def make_dace_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    auto_optimize: bool = False,
    cached_translation: bool = False,
    cmake_build_type: config.CMakeBuildType | None = None,
    translation: definitions.TranslationStep | None = None,
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject] | None = None,
    compilation: workflow.Workflow[stages.CompilableProject, stages.ExecutableProgram]
    | None = None,
    decoration: workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram] | None = None,
) -> recipes.OTFCompileWorkflow:
    """Build the DaCe translation -> bindings -> compilation -> decoration workflow.

    Cross-cutting configuration (device, auto-optimize, translation caching, build
    type) is passed as keyword arguments. To customize translator-local options,
    inject a pre-built translator, e.g.
    ``translation=make_dace_translator(async_sdfg_call=True)``; its ``device_type``
    and ``auto_optimize`` are set to match the cross-cutting arguments.
    """
    cmake_build_type = config.CMAKE_BUILD_TYPE if cmake_build_type is None else cmake_build_type

    bare_translation = workflow.with_fields(
        translation
        if translation is not None
        else make_dace_translator(auto_optimize=auto_optimize),
        device_type=device_type,
        auto_optimize=auto_optimize,
    )
    translation_step: definitions.TranslationStep
    if cached_translation:
        translation_step = workflow.CachedStep.persistent(
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
        translation_step = bare_translation

    return recipes.OTFCompileWorkflow(
        translation=translation_step,
        bindings=bindings
        if bindings is not None
        else functools.partial(
            bindings_step.bind_sdfg, bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME
        ),
        compilation=compilation
        if compilation is not None
        else DaCeCompiler(
            bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME,
            cache_lifetime=config.BUILD_CACHE_LIFETIME,
            device_type=device_type,
            cmake_build_type=cmake_build_type,
        ),
        decoration=decoration
        if decoration is not None
        else functools.partial(decoration_step.convert_args, device=device_type),
    )
