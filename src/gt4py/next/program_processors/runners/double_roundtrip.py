# GT4Py - GridTools Framework
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

from typing import Any

from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.program_processors.runners import roundtrip


@program_executor
def executor(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
    roundtrip.executor(program, *args, dispatch_backend=roundtrip.executor, **kwargs)
