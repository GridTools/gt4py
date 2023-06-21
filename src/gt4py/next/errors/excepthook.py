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

import sys
from typing import Callable

from . import exceptions, formatting


def compilation_error_hook(fallback: Callable, type_: type, value: BaseException, tb):
    if isinstance(value, exceptions.CompilerError):
        print(
            "".join(
                formatting.format_compilation_error(
                    type(value), value.message, value.location_trace
                )
            ),
            file=sys.stderr,
        )
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
