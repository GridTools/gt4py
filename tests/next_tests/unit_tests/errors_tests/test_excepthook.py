# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
