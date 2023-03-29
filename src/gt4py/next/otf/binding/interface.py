# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import gt4py.next.type_system.type_specifications as ts
from gt4py.eve import codegen
from gt4py.next.otf import languages


def format_source(settings: languages.LanguageSettings, source):
    return codegen.format_source(settings.formatter_key, source, style=settings.formatter_style)


@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    type_: ts.TypeSpec


@dataclasses.dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[Parameter, ...]


@dataclasses.dataclass(frozen=True)
class LibraryDependency:
    name: str
    version: str
