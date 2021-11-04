#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
from dataclasses import dataclass, field


class SingleStaticAssignPass(ast.NodeTransformer):
    class RhsRenamer(ast.NodeTransformer):
        """
        Rename right hand side names.

        Only read from parent visitor state, can not modify.
        """

        def __init__(self, state):
            super().__init__()
            self.state = SingleStaticAssignPass.State(name_counter=state.name_counter.copy())

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in self.state.name_counter:
                node.id = f"{node.id}${self.state.name_counter[node.id]}"
            return node

    @dataclass
    class State:
        name_counter: dict[str, int] = field(default_factory=dict)

    def __init__(self):
        super().__init__()
        self.state = self.State()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        # first update rhs names to reference the latest version
        node.value = self.RhsRenamer(self.state).visit(node.value)
        # then update lhs to create new names
        node.targets = [self.visit(target) for target in node.targets]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.state.name_counter:
            self.state.name_counter[node.id] += 1
            node.id = f"{node.id}${self.state.name_counter[node.id]}"
        else:
            self.state.name_counter[node.id] = 0
            node.id = f"{node.id}$0"
        return node
