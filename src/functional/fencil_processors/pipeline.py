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
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

from functional.fencil_processors.builders import cache
from functional.fencil_processors.source_modules import source_modules
from functional.iterator import ir as itir


SrcL = TypeVar("SrcL", bound=source_modules.LanguageTag, covariant=True)
TgtL = TypeVar("TgtL", bound=source_modules.LanguageTag, covariant=True)
LS = TypeVar("LS", bound=source_modules.LanguageSettings, covariant=True)

StartT = TypeVar("StartT")
StartT_contra = TypeVar("StartT_contra", contravariant=True)
EndT = TypeVar("EndT")
EndT_co = TypeVar("EndT_co", covariant=True)
NewEndT = TypeVar("NewEndT")
IntermediateT = TypeVar("IntermediateT")


@dataclasses.dataclass
class OTFFencil:
    fencil: itir.FencilDefinition
    args: list[Any]
    kwargs: dict[str, Any]


class OTFStep(Protocol[StartT_contra, EndT_co]):
    def __call__(self, inp: StartT_contra) -> EndT_co:
        ...


@dataclasses.dataclass
class OTFWorkflow(Generic[StartT, IntermediateT, EndT]):
    first: OTFStep[StartT, IntermediateT]
    second: OTFStep[IntermediateT, EndT]

    def __call__(self, inp: StartT) -> EndT:
        return self.second(self.first(inp))

    def add_step(self, step: OTFStep[EndT, NewEndT]) -> OTFWorkflow[StartT, EndT, NewEndT]:
        return OTFWorkflow(first=self, second=step)


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


class OTFBuilderGenerator(Protocol[SrcL, LS, TgtL]):
    def __call__(
        self, jit_module: source_modules.OTFSourceModule, cache_strategy: cache.Strategy
    ) -> BuildableProject:
        ...


class OTFBuilder(Protocol[SrcL, LS, TgtL]):
    def build(self):
        ...
