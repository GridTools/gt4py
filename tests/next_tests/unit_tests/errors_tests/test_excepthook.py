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

from gt4py import eve
from gt4py.next import errors


def test_format_uncaught_error():
    try:
        loc = eve.SourceLocation("/src/file.py", 1, 1)
        msg = "compile error msg"
        raise errors.exceptions.DSLError(loc, msg) from ValueError("value error msg")
    except errors.exceptions.DSLError as err:
        str_devmode = "".join(errors.excepthook._format_uncaught_error(err, True))
        assert str_devmode.find("Source location") >= 0
        assert str_devmode.find("Traceback") >= 0
        assert str_devmode.find("cause") >= 0
        assert str_devmode.find("ValueError") >= 0
        str_usermode = "".join(errors.excepthook._format_uncaught_error(err, False))
        assert str_usermode.find("Source location") >= 0
        assert str_usermode.find("Traceback") < 0
        assert str_usermode.find("cause") < 0
        assert str_usermode.find("ValueError") < 0
