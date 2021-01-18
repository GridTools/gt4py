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

import re
from typing import Pattern, Union


def match(value: str, regexp: "Union[str, Pattern]") -> bool:
    """
    Check that `regex` pattern matches `value`.

    Stolen from `pytest.raises`.
    Check whether the regular expression `regexp` matches `value` using :func:`python:re.search`.
    If it matches `True` is returned.
    If it doesn't match an `AssertionError` is raised.
    """
    assert re.search(regexp, str(value)), "Pattern {!r} does not match {!r}".format(
        regexp, str(value)
    )
    return True
