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

import inspect

from gt4py.eve import SourceLocation
from gt4py.next.errors import CompilerError


frameinfo = inspect.getframeinfo(inspect.currentframe())
loc = SourceLocation(
    frameinfo.filename, frameinfo.lineno, 1, end_line=frameinfo.lineno, end_column=5
)
raise CompilerError(loc, "this is an error message") from ValueError("asd")
