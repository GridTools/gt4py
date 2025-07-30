# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next.errors import DSLError, excepthook


def test_format_uncaught_error():
    try:
        loc = eve.SourceLocation("/src/file.py", 1, 1)
        msg = "compile error msg"
        raise DSLError(loc, msg) from ValueError("value error msg")
    except DSLError as err:
        str_devmode = "".join(excepthook._format_uncaught_error(err, True))
        assert str_devmode.find("Source location") >= 0
        assert str_devmode.find("Traceback") >= 0
        assert str_devmode.find("cause") >= 0
        assert str_devmode.find("ValueError") >= 0
        str_usermode = "".join(excepthook._format_uncaught_error(err, False))
        assert str_usermode.find("Source location") >= 0
        assert str_usermode.find("Traceback") < 0
        assert str_usermode.find("cause") < 0
        assert str_usermode.find("ValueError") < 0
