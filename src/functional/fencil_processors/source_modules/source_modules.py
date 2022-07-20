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
from typing import Sequence

import numpy


@dataclass
class ScalarParameter:
    name: str
    scalar_type: numpy.dtype


@dataclass
class BufferParameter:
    name: str
    dimensions: Sequence[str]
    scalar_type: numpy.dtype


@dataclass
class Function:
    name: str
    parameters: Sequence[ScalarParameter | BufferParameter]


@dataclass
class LibraryDependency:
    name: str
    version: str


@dataclass
class SourceModule:
    entry_point: Function
    source_code: str
    library_deps: Sequence[LibraryDependency]
    language: str


@dataclass
class BindingModule:
    source_code: str
    library_deps: Sequence[LibraryDependency]
