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
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import eve.codegen

from .builders.cache import Strategy as CacheStrategy
from .source_modules.source_modules import BindingModule, SourceModule


class SupportedLanguageProtocol(Protocol):
    def format_source(self, source_code: str) -> str:
        ...


@dataclass(frozen=True)
class SupportedLanguage(SupportedLanguageProtocol):
    name: str
    implementation_extension: str
    include_extension: str


@dataclass(frozen=True)
class CppLanguage(SupportedLanguage):
    formatting_style: str

    def format_source(self, source_code: str) -> str:
        return eve.codegen.format_source(self.name, source_code, style=self.formatting_style)


CPP_DEFAULT = CppLanguage(
    name="cpp",
    implementation_extension=".cpp",
    include_extension=".cpp.inc",
    formatting_style="LLVM",
)


class BindingsGenerator(Protocol):
    def __call__(
        self, source_module: SourceModule, supported_language: SupportedLanguage
    ) -> BindingModule:
        ...


class BuildProject(Protocol):
    def __init__(
        self,
        source_module: SourceModule,
        bindings_module: Optional[BindingModule],
        cache_strategy: CacheStrategy,
    ):
        ...

    def get_fencil_impl(self) -> Callable:
        ...


class BuildProjectGenerator(Protocol):
    def __call__(
        self,
        source_module: SourceModule,
        bindings_module: Optional[BindingModule],
        language: SupportedLanguage,
    ) -> BuildProject:
        ...
