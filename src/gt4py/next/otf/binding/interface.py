# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

import gt4py.next.type_system.type_specifications as ts
from gt4py.eve import codegen
from gt4py.next.otf import languages


def format_source(settings: languages.LanguageSettings, source: str) -> str:
    return codegen.format_source(settings.formatter_key, source, style=settings.formatter_style)


@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    type_: ts.TypeSpec


@dataclasses.dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[Parameter, ...]
    returns: bool = False


@dataclasses.dataclass(frozen=True)
class LibraryDependency:
    name: str
    version: str
