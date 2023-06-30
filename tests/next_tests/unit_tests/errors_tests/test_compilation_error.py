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

from gt4py.eve import SourceLocation
from gt4py.next.errors import CompilerError


loc = SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)
msg = "a message"


def test_message():
    assert CompilerError(loc, msg).message == msg


def test_location():
    assert CompilerError(loc, msg).location == loc
