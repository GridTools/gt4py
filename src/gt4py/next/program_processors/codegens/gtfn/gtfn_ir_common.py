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

import gt4py.eve as eve
from gt4py.eve import Coerced, SymbolName, SymbolRef


@eve.utils.noninstantiable
class Node(eve.Node):
    pass


class Sym(Node):  # helper
    id: Coerced[SymbolName]  # noqa: A003


class Expr(Node):
    ...


class SymRef(Expr):
    id: Coerced[SymbolRef]  # noqa: A003
