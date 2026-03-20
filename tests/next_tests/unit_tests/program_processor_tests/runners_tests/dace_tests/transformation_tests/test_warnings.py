# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import warnings

from gt4py.next import config as gtx_config


def test_if_warning_is_raised():
    assert not gtx_config.SKIP_WARNINGS, "Tests do not run in debug mode."

    warn_msg = "This is a warning."
    with pytest.warns(UserWarning, match=warn_msg):
        warnings.warn(warn_msg, UserWarning)
