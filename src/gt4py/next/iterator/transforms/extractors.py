# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_specifications as ts


class SymbolNameSetExtractor(eve.NodeVisitor):
    """Extract a set of symbol names"""

    def visit_Literal(self, node: itir.Literal) -> set[str]:
        return set()

    def generic_visitor(self, node: itir.Node) -> set[str]:
        input_fields: set[str] = set()
        for child in eve.trees.iter_children_values(node):
            input_fields |= self.visit(child)
        return input_fields

    def visit_Node(self, node: itir.Node) -> set[str]:
        return set()

    def visit_Program(self, node: itir.Program) -> set[str]:
        names = set()
        for stmt in node.body:
            names |= self.visit(stmt)
        return names

    def visit_IfStmt(self, node: itir.IfStmt) -> set[str]:
        names = set()
        for stmt in node.true_branch + node.false_branch:
            names |= self.visit(stmt)
        return names

    def visit_Temporary(self, node: itir.Temporary) -> set[str]:
        return set()

    def visit_SymRef(self, node: itir.SymRef) -> set[str]:
        return {str(node.id)}

    @classmethod
    def only_fields(cls, program: itir.Program) -> set[str]:
        field_param_names = [
            str(param.id) for param in program.params if isinstance(param.type, ts.FieldType)
        ]
        return {name for name in cls().visit(program) if name in field_param_names}


class InputNamesExtractor(SymbolNameSetExtractor):
    """Extract the set of symbol names passed into field operators within a program."""

    def visit_SetAt(self, node: itir.SetAt) -> set[str]:
        return self.visit(node.expr)

    def visit_FunCall(self, node: itir.FunCall) -> set[str]:
        input_fields = set()
        for arg in node.args:
            input_fields |= self.visit(arg)
        return input_fields


class OutputNamesExtractor(SymbolNameSetExtractor):
    """Extract the set of symbol names written to within a program"""

    def visit_SetAt(self, node: itir.SetAt) -> set[str]:
        return self.visit(node.target)
