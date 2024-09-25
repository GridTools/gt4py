# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py.next import backend as next_backend
from gt4py.next.program_processors.runners import roundtrip


backend = next_backend.Backend(
    transforms=next_backend.DEFAULT_TRANSFORMS,
    executor=roundtrip.RoundtripExecutorFactory(dispatch_backend=roundtrip.default.executor),
    allocator=roundtrip.default.allocator,
)
