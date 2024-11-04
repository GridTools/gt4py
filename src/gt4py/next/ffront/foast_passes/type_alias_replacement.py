# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, cast

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.eve.concepts import SourceLocation, SymbolName, SymbolRef
from gt4py.next.ffront import dialect_ast_enums
from gt4py.next.ffront.fbuiltins import TYPE_BUILTIN_NAMES
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.type_system.type_translation import from_type_hint


@dataclass
class TypeAliasReplacement(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace Type Aliases with their actual type.

    After this pass, the type aliases used for explicit construction of literal
    values and for casting field values are replaced by their actual types.
    """

    closure_vars: dict[str, Any]

    @classmethod
    def apply(
        cls, node: foast.FunctionDefinition | foast.FieldOperator, closure_vars: dict[str, Any]
    ) -> tuple[foast.FunctionDefinition, dict[str, Any]]:
        foast_node = cls(closure_vars=closure_vars).visit(node)
        new_closure_vars = closure_vars.copy()
        for key, value in closure_vars.items():
            if isinstance(value, type) and key not in TYPE_BUILTIN_NAMES:
                new_closure_vars[value.__name__] = closure_vars[key]
        return foast_node, new_closure_vars

    def is_type_alias(self, node_id: SymbolName | SymbolRef) -> bool:
        return (
            node_id in self.closure_vars
            and isinstance(self.closure_vars[node_id], type)
            and node_id not in TYPE_BUILTIN_NAMES
        )

    def visit_Name(self, node: foast.Name, **kwargs: Any) -> foast.Name:
        if self.is_type_alias(node.id):
            return foast.Name(
                id=self.closure_vars[node.id].__name__, location=node.location, type=node.type
            )
        return node

    def _update_closure_var_symbols(
        self, closure_vars: list[foast.Symbol], location: SourceLocation
    ) -> list[foast.Symbol]:
        new_closure_vars: list[foast.Symbol] = []
        existing_type_names: set[str] = set()

        for var in closure_vars:
            if self.is_type_alias(var.id):
                actual_type_name = self.closure_vars[var.id].__name__
                # Avoid multiple definitions of a type in closure_vars
                if actual_type_name not in existing_type_names:
                    new_closure_vars.append(
                        foast.Symbol(
                            id=actual_type_name,
                            type=ts.FunctionType(
                                pos_or_kw_args={},
                                kw_only_args={},
                                pos_only_args=[ts.DeferredType(constraint=ts.ScalarType)],
                                returns=cast(
                                    ts.DataType, from_type_hint(self.closure_vars[var.id])
                                ),
                            ),
                            namespace=dialect_ast_enums.Namespace.CLOSURE,
                            location=location,
                        )
                    )
                    existing_type_names.add(actual_type_name)
            elif var.id not in existing_type_names:
                new_closure_vars.append(var)
                existing_type_names.add(var.id)

        return new_closure_vars

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=self.visit(node.body, **kwargs),
            closure_vars=self._update_closure_var_symbols(node.closure_vars, node.location),
            location=node.location,
        )
