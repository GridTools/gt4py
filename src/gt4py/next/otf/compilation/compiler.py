# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import pathlib
from typing import Protocol, TypeVar

from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import build_data, cache, importer
from gt4py.next.otf.step_types import LS, SrcL, TgtL


SourceLanguageType = TypeVar("SourceLanguageType", bound=languages.LanguageTag)
LanguageSettingsType = TypeVar("LanguageSettingsType", bound=languages.LanguageSettings)
T = TypeVar("T")


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


class BuildSystemProjectGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self,
        source: stages.CompilableSource[SrcL, LS, TgtL],
        cache_strategy: cache.Strategy,
    ) -> stages.BuildSystemProject[SrcL, LS, TgtL]:
        ...


@dataclasses.dataclass(frozen=True)
class Compiler(
    step_types.CompilationStep[SourceLanguageType, LanguageSettingsType, languages.Python]
):
    """Use any build system (via configured factory) to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_strategy: cache.Strategy
    builder_factory: BuildSystemProjectGenerator[
        SourceLanguageType, LanguageSettingsType, languages.Python
    ]
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableSource[SourceLanguageType, LanguageSettingsType, languages.Python],
    ) -> stages.CompiledProgram:
        src_dir = cache.get_cache_folder(inp, self.cache_strategy)

        data = build_data.read_data(src_dir)

        if not data or not is_compiled(data) or self.force_recompile:
            self.builder_factory(inp, self.cache_strategy).build()

        new_data = build_data.read_data(src_dir)

        if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
            raise CompilationError(
                "On-the-fly compilation unsuccessful for {inp.source_module.entry_point.name}!"
            )

        return getattr(
            importer.import_from_path(src_dir / new_data.module), new_data.entry_point_name
        )

    def chain(
        self, step: workflow.Workflow[stages.CompiledProgram, T]
    ) -> workflow.CombinedStep[
        stages.CompilableSource[SourceLanguageType, LanguageSettingsType, languages.Python],
        stages.CompiledProgram,
        T,
    ]:
        return workflow.CombinedStep(first=self, second=step)


class CompilationError(RuntimeError):
    ...
