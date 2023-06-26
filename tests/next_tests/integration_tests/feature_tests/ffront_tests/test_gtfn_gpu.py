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

import pytest

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn_gpu

from next_tests.integration_tests.feature_tests import cases
from next_tests.integration_tests.feature_tests.cases import cartesian_case, no_default_backend
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("fieldview_backend", [gtfn_gpu.gtfn_gpu])
def test_copy(cartesian_case):  # noqa: F811 # fixtures
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)
