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
from typing import Callable, Optional, Protocol, TypeVar

from functional.fencil_processors.builders import cache
from functional.fencil_processors.source_module import source_modules


LanguageT_co = TypeVar("LanguageT_co", bound=source_modules.SupportedLanguage, covariant=True)
LanguageT_contra = TypeVar(
    "LanguageT_contra", bound=source_modules.SupportedLanguage, contravariant=True
)


class BindingsGenerator(Protocol[LanguageT_co]):
    def __call__(
        self, source_module: source_modules.SourceModule[LanguageT_co]
    ) -> source_modules.BindingModule:
        ...


class BuildProject(Protocol):
    def get_fencil_impl(self) -> Callable:
        ...


class BuildProjectGenerator(Protocol[LanguageT_co]):
    def __call__(
        self,
        source_module: source_modules.SourceModule[LanguageT_co],
        bindings_module: Optional[source_modules.BindingModule[LanguageT_co]],
        cache_strategy: cache.Strategy,
    ) -> BuildProject:
        ...
