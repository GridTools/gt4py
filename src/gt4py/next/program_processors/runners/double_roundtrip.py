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

from gt4py.next import backend as next_backend, backend_exp
from gt4py.next.program_processors.runners import roundtrip


backend = backend_exp.ExpBackend(
    transforms_fop=next_backend.DEFAULT_TRANSFORMS,
    transforms_prog=next_backend.DEFAULT_TRANSFORMS,
    executor=roundtrip.RoundtripExecutorFactory(dispatch_backend=roundtrip.default.executor),
    allocator=roundtrip.default.allocator,
)
