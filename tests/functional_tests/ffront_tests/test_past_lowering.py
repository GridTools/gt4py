# GT4Py Project - GridTools Framework
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
from typing import Tuple

import pytest

from functional.common import Field, GridType
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import Dimension, float64
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_to_itir import ProgramLowering


IDim = Dimension("[IDim]")


@field_operator
def make_tuple_op(
    inp: Field[[IDim], float64]
) -> Tuple[Field[[IDim], float64], Field[[IDim], float64]]:
    return inp, inp


def test_tuple_constructed_in_out_invalid_slicing():
    def tuple_program(
        inp: Field[[IDim], float64], out1: Field[[IDim], float64], out2: Field[[IDim], float64]
    ):
        make_tuple_op(inp, out=(out1[1:], out2))

    parsed = ProgramParser.apply_to_function(tuple_program)
    with pytest.raises(
        RuntimeError,
        match="Unexpected `out` argument",
    ):
        ProgramLowering.apply(parsed, function_definitions=[], grid_type=GridType.CARTESIAN)
