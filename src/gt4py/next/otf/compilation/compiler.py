# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
from typing import Protocol, TypeVar

import factory
from flufl import lock

from gt4py.next import config
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.compilation import build_data, cache, importer
from gt4py.next.otf.step_types import LS, SrcL, TgtL


SourceLanguageType = TypeVar("SourceLanguageType", bound=languages.NanobindSrcL)
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
        cache_lifetime: config.BuildCacheLifetime,
    ) -> stages.BuildSystemProject[SrcL, LS, TgtL]: ...


@dataclasses.dataclass(frozen=True)
class Compiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[SourceLanguageType, LanguageSettingsType, languages.Python],
        stages.CompiledProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[SourceLanguageType, LanguageSettingsType, languages.Python],
        stages.CompiledProgram,
    ],
    step_types.CompilationStep[SourceLanguageType, LanguageSettingsType, languages.Python],
):
    """Use any build system (via configured factory) to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[
        SourceLanguageType, LanguageSettingsType, languages.Python
    ]
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableSource[SourceLanguageType, LanguageSettingsType, languages.Python],
    ) -> stages.ExtendedCompiledProgram:
        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)

        # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
        # we need to make sure that only one of them accesses the same build directory for compilation.
        with lock.Lock(str(src_dir / "compilation.lock"), lifetime=600):  # type: ignore[attr-defined] # mypy not smart enough to understand custom export logic
            data = build_data.read_data(src_dir)

            if not data or not is_compiled(data) or self.force_recompile:
                self.builder_factory(inp, self.cache_lifetime).build()

            new_data = build_data.read_data(src_dir)

            if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
                raise CompilationError(
                    f"On-the-fly compilation unsuccessful for '{inp.program_source.entry_point.name}'."
                )

        compiled_prog = getattr(
            importer.import_from_path(src_dir / new_data.module), new_data.entry_point_name
        )

        @dataclasses.dataclass(frozen=True)
        class Wrapper(stages.ExtendedCompiledProgram):
            implicit_domain: bool = inp.program_source.implicit_domain
            __call__: stages.CompiledProgram = compiled_prog

        return Wrapper()


class CompilerFactory(factory.Factory):
    class Meta:
        model = Compiler


class CompilationError(RuntimeError): ...
