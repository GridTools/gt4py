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

from typing import TYPE_CHECKING, Any

import gt4py.next.program_processors.processor_interface as ppi
from gt4py.next import backend as next_backend
from gt4py.next.program_processors.runners import roundtrip
from gt4py.next.program_processors import modular_executor
from gt4py.next.otf import transforms

if TYPE_CHECKING:
    import gt4py.next.iterator.ir as itir


#@ppi.program_executor
def executor() -> None:
    #roundtrip.execute_roundtrip(program, *args, dispatch_backend=roundtrip.executor, **kwargs)
    modular_executor.ModularExecutor(
        otf_workflow=transforms.PastToItir().chain(roundtrip.ExecuteRoundtrip()),
        name="roundtrip"
    )
backend = next_backend.Backend(
    executor=executor,
    allocator=roundtrip.backend.allocator,
)
