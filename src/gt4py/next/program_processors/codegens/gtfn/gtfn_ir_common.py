# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.eve as eve
from gt4py.eve import Coerced, SymbolName, SymbolRef


@eve.utils.noninstantiable
class Node(eve.Node):
    pass


class Sym(Node):  # helper
    id: Coerced[SymbolName]


class Expr(Node): ...


class SymRef(Expr):
    id: Coerced[SymbolRef]
