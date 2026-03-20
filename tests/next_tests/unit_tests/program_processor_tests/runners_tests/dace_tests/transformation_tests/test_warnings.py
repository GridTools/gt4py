# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import warnings

# Without this the test fails if DaCe is not installed, even when the `requires_dace`
#  marker is configured in `__init__.py`.
dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


def test_if_warning_is_raised():
    warn_msg = "This is a warning."

    with pytest.warns(UserWarning, match=warn_msg):
        warnings.warn(warn_msg, UserWarning)
