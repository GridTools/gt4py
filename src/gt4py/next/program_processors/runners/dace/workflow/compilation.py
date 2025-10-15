# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import dace
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.eve import utils as eve_utils
from gt4py.next import common, config, utils as gtx_utils
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace import sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


def _hash_call_args(args: Sequence[Any]) -> int:
    return hash(
        gtx_utils.tree_map(
            lambda x: x
            if (ndarray := getattr(x, "ndarray", None)) is None
            else (x.data_ptr(), x.domain, ndarray.strides)
        )(tuple(args)),
    )


class CompiledDaceProgram(stages.CompiledProgram):
    sdfg_program: dace.CompiledSDFG
    call_args_cache: eve_utils.CustomMapping

    def __init__(self, program: dace.CompiledSDFG):
        self.sdfg_program = program
        self.call_args_cache = eve_utils.CustomMapping(_hash_call_args)

    def _construct_args(
        self,
        offset_provider: common.OffsetProvider,
        *args: Sequence[Any],
    ) -> tuple[tuple[Any], tuple[Any]]:
        kwargs = sdfg_callable.get_sdfg_args(
            self.sdfg_program.sdfg,
            offset_provider,
            *args,
            filter_args=False,
        )
        kwargs[gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL] = config.COLLECT_METRICS_LEVEL
        return self.sdfg_program._construct_args(kwargs)

    def __call__(
        self,
        offset_provider: common.OffsetProvider,
        *args: Sequence[Any],
    ) -> Any:
        try:
            argtuple, initargtuple = self.call_args_cache[args]
        except KeyError:
            argtuple, initargtuple = self._construct_args(offset_provider, *args)
            self.call_args_cache[args] = (argtuple, initargtuple)
        return self.sdfg_program.fast_call(argtuple, initargtuple, do_gpu_check=True)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    step_types.CompilationStep[languages.SDFG, languages.LanguageSettings, languages.Python],
):
    """Use the dace build system to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> CompiledDaceProgram:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = gtx_cache.get_cache_folder(inp, self.cache_lifetime)
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg.build_folder = sdfg_build_folder
            with locking.lock(sdfg_build_folder):
                sdfg_program = sdfg.compile(validate=False)

        return CompiledDaceProgram(sdfg_program)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
