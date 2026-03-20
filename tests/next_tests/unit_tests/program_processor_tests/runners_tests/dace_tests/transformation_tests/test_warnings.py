# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import copy


def test_if_warning_is_raised():
    # Avoid importing DaCe as a side effect during test discovery if DaCe is not installed.
    from gt4py.next.program_processors.runners.dace import (
        transformations as gtx_transformations,
    )

    warn_msg = "This is a warning."

    with pytest.warns(UserWarning, match=warn_msg):
        gtx_transformations.utils.warn(warn_msg, UserWarning)
