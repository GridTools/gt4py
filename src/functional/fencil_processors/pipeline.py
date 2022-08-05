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
from pathlib import Path
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

    def find_fs_cache(self) -> Path:
        ...

    # write unconditionally
    def write(self) -> None:
        ...

    # trigger underlying configure unconditionally, fail if not written to fs
    def configure(self) -> None:
        ...

    def is_configured(self) -> bool:
        ...

    # trigger underlying build unconditionally but fail if not configured
    def build(self) -> None:
        # TODO(ricoh): encode the requirement that every build output file has a unique name
        #   because python can not re-import a module cleanly. Best put the hash into the output name,
        #   then the build system's build command only needs to be triggered if the hash changes.
        #   Otherwise the fencil function wouldn't change anyway so why bother.
        #   (https://docs.python.org/3.4/library/importlib.html#importlib.reload does not recommend
        #   importlib.reload() for dynamically imported modules, especially extensions.)
        ...

    def is_built(self) -> bool:
        ...

    # trigger whatever steps are necessary, use caches when safely possible
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
