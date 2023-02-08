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

from typing import Protocol, TypeVar

from gt4py.next.otf import languages, stages


SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
LS = TypeVar("LS", bound=languages.LanguageSettings)
SrcL_co = TypeVar("SrcL_co", bound=languages.LanguageTag, covariant=True)
TgtL_co = TypeVar("TgtL_co", bound=languages.LanguageTag, covariant=True)
LS_co = TypeVar("LS_co", bound=languages.LanguageSettings, covariant=True)


class TranslationStep(Protocol[SrcL, LS]):
    """Translate a GT4Py program to source code (ProgramCall -> ProgramSource)."""

    def __call__(self, program_call: stages.ProgramCall) -> stages.ProgramSource[SrcL, LS]:
        ...


class BindingStep(Protocol[SrcL, LS, TgtL]):
    """
    Generate Bindings for program source and package both together (ProgramSource -> CompilableSource).

    In the special cases where bindings are not required, such a step could also simply construct
    a ``CompilableSource`` from the ``ProgramSource`` with bindings set to ``None``.
    """

    def __call__(
        self, program_source: stages.ProgramSource[SrcL, LS]
    ) -> stages.CompilableSource[SrcL, LS, TgtL]:
        ...


class CompilationStep(Protocol[SrcL, LS, TgtL]):
    """Compile program source code and bindings into a python callable (CompilableSource -> CompiledProgram)."""

    def __call__(self, source: stages.CompilableSource[SrcL, LS, TgtL]) -> stages.CompiledProgram:
        ...
