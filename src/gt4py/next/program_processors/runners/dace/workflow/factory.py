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

import factory

from gt4py._core import definitions as core_defs, filecache
from gt4py.next import config
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.program_processors.runners.dace.workflow import (
    bindings as bindings_step,
    decoration as decoration_step,
)
from gt4py.next.program_processors.runners.dace.workflow.compilation import (
    DaCeCompilationStepFactory,
)
from gt4py.next.program_processors.runners.dace.workflow.translation import (
    DaCeTranslationStepFactory,
)


_GT_DACE_BINDING_FUNCTION_NAME: Final[str] = "update_sdfg_args"


class DaCeWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        auto_optimize: bool = False
        make_persistent: bool = False
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(  # type: ignore[assignment] # factory-boy typing not precise enough
            lambda: config.CMAKE_BUILD_TYPE
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=stages.fingerprint_compilable_program,
                    cache=filecache.FileCache(str(config.BUILD_CACHE_DIR / "translation_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            DaCeTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
            make_persistent=factory.SelfAttribute("..make_persistent"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)
    bindings = factory.LazyAttribute(
        lambda o: functools.partial(
            bindings_step.bind_sdfg,
            bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME,
            make_persistent=o.make_persistent,
        )
    )
    compilation = factory.SubFactory(
        DaCeCompilationStepFactory,
        bind_func_name=_GT_DACE_BINDING_FUNCTION_NAME,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        device_type=factory.SelfAttribute("..device_type"),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            decoration_step.convert_args,
            device=o.device_type,
        )
    )
