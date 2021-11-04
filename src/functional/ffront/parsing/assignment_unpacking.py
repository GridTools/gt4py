#!/usr/bin/env python
# -*- coding: utf-8 -*-


import ast


class NodeYielder(ast.NodeTransformer):
    def visit(self, node: ast.AST) -> Iterator[ast.AST]:
        yield from super().visit(node)

    def generic_visit(self, node: ast.AST) -> Iterator[ast.AST]:



class AssignmentUnpacker(ast.NodeVisitor):

