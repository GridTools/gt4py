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

import os

from gt4py import eve
from gt4py.next import errors
from gt4py.next.errors import excepthook


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


def test_get_verbose_exceptions():
    env_var_name = "GT4PY_VERBOSE_EXCEPTIONS"

    # Make sure to save and restore the environment variable, we don't want to
    # affect other tests running in the same process.
    saved = os.environ.get(env_var_name, None)
    try:
        os.environ[env_var_name] = "False"
        assert excepthook._get_verbose_exceptions_envvar() is False
        os.environ[env_var_name] = "True"
        assert excepthook._get_verbose_exceptions_envvar() is True
        os.environ[env_var_name] = "invalid value"  # Should emit a warning too
        assert excepthook._get_verbose_exceptions_envvar() is False
        del os.environ[env_var_name]
        assert excepthook._get_verbose_exceptions_envvar() is False
    finally:
        if saved is not None:
            os.environ[env_var_name] = saved
