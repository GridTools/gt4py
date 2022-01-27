# GT4Py Project - GridTools Framework
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
from typing import Union

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, SymbolTableTrait


class FieldOperatorShiftRecognition(NodeTranslator):
    """Transform `field(Offset[value])` syntax into shifts."""

    context = (SymbolTableTrait.symtable_merger,)

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_Call(self, node: foast.Call, **kwargs) -> Union[foast.Call, foast.Shift]:
        if isinstance(node.func.type, foast.FieldType):
            result = foast.Shift(
                offsets=node.args, expr=node.func, location=node.location, type=node.func.type
            )
            return result
        return node
