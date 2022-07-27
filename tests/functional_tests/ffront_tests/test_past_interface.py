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

from functional.common import Field
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import float64
from functional.ffront.func_to_past import ProgramParser


@field_operator
def make_tuple_op(inp: Field[..., float64]) -> Tuple[Field[..., float64], Field[..., float64]]:
    return inp, inp


# --- Parsing ---
def test_tuple_constructed_in_out():
    def tuple_program(
        inp: Field[..., float64], out1: Field[..., float64], out2: Field[..., float64]
    ):
        make_tuple_op(inp, out=(out1, out2))

    _ = ProgramParser.apply_to_function(tuple_program)
