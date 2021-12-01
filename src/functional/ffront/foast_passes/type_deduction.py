#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
import warnings
from typing import Optional

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator


def is_complete_type(fo_type: foast.DataType) -> bool:
    if fo_type is None:
        return False
    elif isinstance(fo_type, foast.DeferredSymbolType):
        return False
    elif isinstance(fo_type, foast.SymbolType):
        return True
    return False


class FieldOperatorTypeDeduction(NodeTranslator):
    """Deduce and check types of FOAST expressions and symbols."""

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        return cls().visit(node)

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        kwargs["symtable"] = kwargs.get("symtable", node.symtable_)
        return foast.FieldOperator(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            body=self.visit(node.body, **kwargs),
            location=node.location,
        )

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs.get("symtable", {})
        if node.id not in symtable:
            warnings.warn(  # TODO (ricoh): raise this instead (requires externals)
                FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg=f"Undeclared symbol {node.id}"
                )
            )
            return node

        symbol = symtable[node.id]
        return foast.Name(id=node.id, type=symbol.type, location=node.location)


class FieldOperatorTypeDeductionError(SyntaxError, SyntaxWarning):
    def __init__(
        self,
        msg="",
        *,
        lineno=0,
        offset=0,
        filename=None,
        end_lineno=None,
        end_offset=None,
        text=None,
    ):
        msg = "Could not deduce type: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_foast_node(
        cls,
        node: foast.LocatedNode,
        *,
        msg: str = "",
        filename: Optional[str] = None,
        text: Optional[str] = None,
    ):
        return cls(
            msg,
            lineno=node.location.line,
            offset=node.location.column,
            filename=node.location.source,
            end_lineno=node.location.end_line,
            end_offset=node.location.end_column,
        )
