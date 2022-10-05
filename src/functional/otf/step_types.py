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

from typing import Protocol, TypeVar

from functional.otf import languages, stages
from functional.program_processors.builders import cache


SrcL = TypeVar("SrcL", bound=languages.LanguageTag)
TgtL = TypeVar("TgtL", bound=languages.LanguageTag)
LS = TypeVar("LS", bound=languages.LanguageSettings)
SrcL_co = TypeVar("SrcL_co", bound=languages.LanguageTag, covariant=True)
TgtL_co = TypeVar("TgtL_co", bound=languages.LanguageTag, covariant=True)
LS_co = TypeVar("LS_co", bound=languages.LanguageSettings, covariant=True)


class BindingsGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self, program_source: stages.ProgramSource[SrcL, LS]
    ) -> stages.BindingSource[SrcL, TgtL]:
        ...


class CompilableSourceGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self, program_source: stages.ProgramSource[SrcL, LS]
    ) -> stages.CompilableSource[SrcL, LS, TgtL]:
        ...


class BuildSystemProjectGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self,
        source: stages.CompilableSource[SrcL, LS, TgtL],
        cache_strategy: cache.Strategy,
    ) -> stages.BuildSystemProject[SrcL, LS, TgtL]:
        ...
