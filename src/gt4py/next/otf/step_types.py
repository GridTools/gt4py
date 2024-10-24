# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Protocol, TypeVar

from gt4py.next.otf import languages, stages, workflow


SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
LS = TypeVar("LS", bound=languages.LanguageSettings)
SrcL_co = TypeVar("SrcL_co", bound=languages.LanguageTag, covariant=True)
TgtL_co = TypeVar("TgtL_co", bound=languages.LanguageTag, covariant=True)
LS_co = TypeVar("LS_co", bound=languages.LanguageSettings, covariant=True)


class TranslationStep(
    workflow.ReplaceEnabledWorkflowMixin[stages.CompilableProgram, stages.ProgramSource[SrcL, LS]],
    Protocol[SrcL, LS],
):
    """Translate a GT4Py program to source code (ProgramCall -> ProgramSource)."""

    ...


class BindingStep(Protocol[SrcL, LS, TgtL]):
    """
    Generate Bindings for program source and package both together (ProgramSource -> CompilableSource).

    In the special cases where bindings are not required, such a step could also simply construct
    a ``CompilableSource`` from the ``ProgramSource`` with bindings set to ``None``.
    """

    def __call__(
        self, program_source: stages.ProgramSource[SrcL, LS]
    ) -> stages.CompilableSource[SrcL, LS, TgtL]: ...


class CompilationStep(
    workflow.Workflow[stages.CompilableSource[SrcL, LS, TgtL], stages.CompiledProgram],
    Protocol[SrcL, LS, TgtL],
):
    """Compile program source code and bindings into a python callable (CompilableSource -> CompiledProgram)."""

    def __call__(
        self, source: stages.CompilableSource[SrcL, LS, TgtL]
    ) -> stages.CompiledProgram: ...
