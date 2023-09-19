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
import enum

import pytest


# Skip definitions
class SkipMark(enum.Enum):
    XFAIL = pytest.xfail
    SKIP = pytest.skip


# Skip messages (available format keys: 'marker', 'backend')
UNSUPPORTED_MESSAGE = "'{marker}' tests not supported by '{backend}' backend"

# Processors
DACE = "gt4py.next.program_processors.runners.dace_iterator.run_dace_iterator"

# Test markers
USES_ORIGIN = "uses_origin"
USES_TUPLE_RETURNS = "uses_tuple_returns"

# Skip matrix
BACKEND_SKIP_TEST_MATRIX = {
    DACE: [
        (USES_ORIGIN, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_TUPLE_RETURNS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
    ]
}


def get_processor_id(processor):
    if hasattr(processor, "__module__") and hasattr(processor, "__name__"):
        module_path = processor.__module__.split(".")[-1]
        name = processor.__name__
        return f"{module_path}.{name}"
    return repr(processor)
