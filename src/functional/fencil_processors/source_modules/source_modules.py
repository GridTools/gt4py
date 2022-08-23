# GT4Py Project - GridTools Framework
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
"""Structures that provide a unified interface for connecting source code generators and builders."""


from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Optional, Protocol, TypeVar

import numpy as np

import eve.codegen


class SupportedLanguage(Protocol):
    """Ensures consistent code formatting along the pipeline."""

    @property
    def name(self) -> str:
        ...

    def format_source(self, source_code: str) -> str:
        ...


class LanguageWithHeaders(SupportedLanguage, Protocol):
    """Ensures consistent file naming for languages that split code into include (header) and implementation files."""

    @property
    def implementation_extension(self) -> str:
        ...

    @property
    def include_extension(self) -> str:
        ...


class LanguageTag:
    settings_level: ClassVar[type[LanguageSettings]]
    ...


SrcL = TypeVar("SrcL", bound=LanguageTag)
TgtL = TypeVar("TgtL", bound=LanguageTag)


@dataclass(frozen=True)
class LanguageSettings:
    formatter_key: str
    formatter_style: Optional[str]
    file_extension: str


SettingT = TypeVar("SettingT", bound=LanguageSettings)


@dataclass(frozen=True)
class LanguageWithHeaderFilesSettings(LanguageSettings):
    header_extension: str


class Python(LanguageTag):
    settings_level = LanguageSettings
    ...


class Cpp(LanguageTag):
    settings_level = LanguageWithHeaderFilesSettings
    ...


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    scalar_type: np.dtype


@dataclass(frozen=True)
class BufferParameter:
    name: str
    dimensions: tuple[str, ...]
    scalar_type: np.dtype


@dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[ScalarParameter | BufferParameter, ...]


@dataclass(frozen=True)
class LibraryDependency:
    name: str
    version: str


@dataclass(frozen=True)
class SourceModule(Generic[SrcL, SettingT]):
    entry_point: Function
    source_code: str
    library_deps: tuple[LibraryDependency, ...]
    language: type[SrcL]
    language_settings: SettingT

    def __post_init__(self):
        if not issubclass(self.__class__, self.language.settings_level):
            raise TypeError(
                f"Wrong language settings type for {self.language}, must be subclass of {self.language.settings_level}"
            )


@dataclass(frozen=True)
class BindingModule(Generic[SrcL, TgtL]):
    source_code: str
    library_deps: tuple[LibraryDependency, ...]


def format_source(settings: LanguageSettings, source):
    return eve.codegen.format_source(settings.formatter_key, source, style=settings.formatter_style)
