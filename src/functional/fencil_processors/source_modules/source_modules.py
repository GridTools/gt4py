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


from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np


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


LanguageT_co = TypeVar("LanguageT_co", bound=SupportedLanguage, covariant=True)
LanguageT_contra = TypeVar("LanguageT_contra", bound=SupportedLanguage, contravariant=True)


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
class SourceModule(Generic[LanguageT_co]):
    entry_point: Function
    source_code: str
    library_deps: tuple[LibraryDependency, ...]
    language: LanguageT_co


@dataclass(frozen=True)
class BindingModule(Generic[LanguageT_contra]):
    source_code: str
    library_deps: tuple[LibraryDependency, ...]
