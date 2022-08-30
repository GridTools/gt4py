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
from functional.fencil_processors.source_modules import source_modules


SrcL = TypeVar("SrcL", bound=source_modules.LanguageTag, covariant=True)
TgtL = TypeVar("TgtL", bound=source_modules.LanguageTag, covariant=True)
LS = TypeVar("LS", bound=source_modules.LanguageSettings, covariant=True)


class BindingsGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self, source_module: source_modules.SourceModule[SrcL, LS]
    ) -> source_modules.BindingModule[SrcL, TgtL]:
        ...


class BuildableProject(Protocol):
    def get_fencil_impl(self) -> Callable:
        ...


class BuildableProjectGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self,
        source_module: source_modules.SourceModule[SrcL, LS],
        bindings_module: Optional[source_modules.BindingModule[SrcL, TgtL]],
        cache_strategy: cache.Strategy,
    ) -> BuildableProject:
        ...
