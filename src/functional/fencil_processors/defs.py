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


from dataclasses import dataclass
from typing import Sequence, Type


@dataclass
class ScalarParameter:
    name: str
    scalar_type: Type


@dataclass
class BufferParameter:
    name: str
    dimensions: Sequence[str]
    scalar_type: Type


@dataclass
class ConnectivityParameter:
    name: str
    offset_tag: str


@dataclass
class Function:
    name: str
    parameters: Sequence[ConnectivityParameter | ScalarParameter | BufferParameter]


@dataclass
class LibraryDependency:
    name: str
    version: str


@dataclass
class SourceCodeModule:
    entry_point: Function
    source_code: str
    library_deps: Sequence[LibraryDependency]
    language: str


@dataclass
class BindingCodeModule:
    source_code: str
    library_deps: Sequence[LibraryDependency]
