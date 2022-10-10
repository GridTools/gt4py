# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from functional.otf import languages, stages, step_types, workflow
from functional.otf.compilation import build_data, cache, importer
from functional.otf.step_types import LS, SrcL, TgtL


SL = TypeVar("SL", bound=languages.LanguageTag)
ST = TypeVar("ST", bound=languages.LanguageSettings)
NT = TypeVar("NT")


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
class Compiler(step_types.CompilationStep[SL, ST, languages.Python]):
    cache_strategy: cache.Strategy
    builder_factory: BuildSystemProjectGenerator[SL, ST, languages.Python]
    force_recompile: bool = False
    """Use any build system (via configured factory) to compile a GT4Py program to a ``functional.otf.stages.CompiledProgram``."""

    def __call__(
        self, inp: stages.CompilableSource[SL, ST, languages.Python]
    ) -> stages.CompiledProgram:
        src_dir = cache.get_cache_folder(inp, self.cache_strategy)

        data = build_data.read_data(src_dir)

        if not data or not is_compiled(data) or self.force_recompile:
            self.builder_factory(inp, self.cache_strategy).build()

        new_data = build_data.read_data(src_dir)

        if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
            raise CompilerError(
                "On-the-fly compilation unsuccessful for {inp.source_module.entry_point.name}!"
            )

        return getattr(
            importer.import_from_path(src_dir / new_data.module), new_data.entry_point_name
        )

    def chain(
        self, step: workflow.StepProtocol[stages.CompiledProgram, NT]
    ) -> workflow.Workflow[
        stages.CompilableSource[SL, ST, languages.Python], stages.CompiledProgram, NT
    ]:
        return workflow.Workflow(first=self, second=step)


class CompilerError(Exception):
    ...
