# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from pydantic.error_wrappers import ValidationError

from gtc.common import DataType
from gtc.oir import Temporary

from .oir_utils import AssignStmtFactory, FieldAccessFactory, HorizontalExecutionFactory


def test_no_horizontal_offset_allowed():
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        AssignStmtFactory(left__offset__i=1)


def test_mask_must_be_bool():
    with pytest.raises(ValidationError, match=r".*must be.* bool.*"):
        HorizontalExecutionFactory(mask=FieldAccessFactory(dtype=DataType.INT32))


def test_temporary_default_3d():
    temp = Temporary(name="a", dtype=DataType.INT64)
    assert temp.dimensions == (True, True, True)

    temp1d = Temporary(name="b", dtype=DataType.INT64, dimensions=(True, False, False))
    assert temp1d.dimensions == (True, False, False)
