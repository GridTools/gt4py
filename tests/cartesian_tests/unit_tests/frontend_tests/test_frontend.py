# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian import frontend


def test_bad_frontend_feedback():
    existing_backend = frontend.from_name("gtscript")
    assert existing_backend

    with pytest.raises(ValueError):
        frontend.from_name("xxxxx")
