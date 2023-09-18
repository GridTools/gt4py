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

from dataclasses import dataclass
from typing import Any

from gt4py.eve.concepts import SourceLocation, SymbolName, SymbolRef
import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.next.ffront import dialect_ast_enums
from gt4py.next.ffront.fbuiltins import TYPE_BUILTIN_NAMES
from gt4py.next.type_system.type_translation import get_scalar_kind
from gt4py.next.type_system import type_specifications as ts

@dataclass
class TypeAliasReplacement(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace Type Aliases with their actual type.

    After this pass, the type aliases used for explicit construction of literal
    values and for casting field values are replaced by their actual types.
    """

    closure_vars: dict[str, Any]

    @classmethod
    def apply(cls, node: foast.FunctionDefinition, closure_vars: dict[str, Any]) -> tuple[
        foast.FunctionDefinition, dict[str, Any]]:
        foast_node = cls(closure_vars=closure_vars).visit(node)
        closures_copy = closure_vars.copy()
        for key, value in closures_copy.items():
            if isinstance(value, type) and key not in TYPE_BUILTIN_NAMES:
                closure_vars[value.__name__] = closure_vars.pop(key)

        return foast_node, closure_vars

    def _get_actual_type_name(self, id: SymbolName | SymbolRef) -> str | SymbolName | SymbolRef:
        if id in self.closure_vars and isinstance(self.closure_vars[id], type) and id not in TYPE_BUILTIN_NAMES:
            return self.closure_vars[id].__name__
        return id

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        actual_type_name = self._get_actual_type_name(node.id)
        new_node_id = actual_type_name if not isinstance(actual_type_name, SymbolRef) else node.id
        return foast.Name(id=new_node_id, location=node.location, type=node.type)

    def _add_actual_type_to_closure_vars(self, var: foast.Symbol, location: SourceLocation) -> foast.Symbol:
        actual_type_name = self._get_actual_type_name(var.id)
        if not isinstance(actual_type_name, SymbolName):
            return foast.Symbol(
                id=actual_type_name,
                type=ts.FunctionType(
                    pos_or_kw_args={},
                    kw_only_args={},
                    pos_only_args=[ts.DeferredType(constraint=ts.ScalarType)],
                    returns=ts.ScalarType(kind=get_scalar_kind(self.closure_vars[var.id]))
                ),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=location
            )
        return var

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs) -> foast.FunctionDefinition:
        new_closure_vars = []

        for var in node.closure_vars:
            actual_type_symbol = self._add_actual_type_to_closure_vars(var, node.location)
            new_closure_vars.append(actual_type_symbol)

        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=self.visit(node.body, **kwargs),
            closure_vars=new_closure_vars,
            location=node.location,
        )
