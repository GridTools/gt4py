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
from typing import Protocol, TypeVar, cast

import factory

from gt4py._core import locking
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
    ) -> stages.CompiledProgram:
        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)

        # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
        # we need to make sure that only one of them accesses the same build directory for compilation.
        with locking.lock(src_dir):
            data = build_data.read_data(src_dir)

            if not data or not is_compiled(data) or self.force_recompile:
                self.builder_factory(inp, self.cache_lifetime).build()

            new_data = build_data.read_data(src_dir)

            if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
                raise CompilationError(
                    f"On-the-fly compilation unsuccessful for '{inp.program_source.entry_point.name}'."
                )

        m = importer.import_from_path(src_dir / new_data.module)
        func = getattr(m, new_data.entry_point_name)

        # Since nanobind 2.10, calling functions with ndarray args crashes (SEGFAULT)
        # when there are not live references to their extension module (see: https://github.com/wjakob/nanobind/issues/1283)
        # Here we dynamically create a new callable class holding a reference to the
        # module and the function, whose `__call__` is exactly the `__call__` method
        # of the returned (nanobind) nbfunction object. As long as this object is alive,
        # the module reference is kept alive too, preventing the SEGFAULT.
        managed_entry_point = type(
            f"{m.__name__}_managed_wrapper",
            (),
            dict(
                __call__=func.__call__,
                __doc__=getattr(func, "__doc__", None),
                __hash__=func.__hash__,
                __eq__=func.__eq__,
                module_ref=m,
                func_ref=func,
            ),
        )()

        return cast(stages.CompiledProgram, managed_entry_point)


class CompilerFactory(factory.Factory):
    class Meta:
        model = Compiler


class CompilationError(RuntimeError): ...
