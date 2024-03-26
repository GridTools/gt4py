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
import types
from typing import Any, Optional

from gt4py.next import common
from gt4py.next.ffront import program_ast as past


@dataclasses.dataclass(frozen=True)
class ProgramDefinition:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastClosure:
    closure_vars: dict[str, Any]
    past_node: past.Program
    grid_type: Optional[common.GridType]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
