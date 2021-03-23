# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

from typing import List

import networkx as nx

import eve  # noqa: F401
from eve import NodeVisitor
from gtc_unstructured.irs.nir import AssignStmt, FieldAccess, HorizontalLoop


class _FieldWriteDependencyGraph(NodeVisitor):
    """Returns a dependency graph of field writes for a list of horizontal loops.

    Result is a DAG where nodes represent writes and edges represent reads with extent information.

    Example 1:
    A = 1
    B = A(with offset)

    Graph: A --has_extent--> B

    Example 2:
    A = B (B is external to the loop)

    Graph: A
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.graph = nx.DiGraph()
        self.last_write_access = {}

    @classmethod
    def generate(cls, loops, **kwargs):
        """Generate dependency graph."""
        instance = cls()
        for loop in loops:
            instance.visit(loop, **kwargs)
        return instance.graph

    def visit_FieldAccess(self, node: FieldAccess, **kwargs):
        assert "current_write" in kwargs
        if node.name in self.last_write_access:
            assert self.last_write_access[node.name] in self.graph.nodes()
            source = self.last_write_access[node.name]
            self.graph.add_edge(source, kwargs["current_write"], extent=node.extent)

    def visit_AssignStmt(self, node: AssignStmt, **kwargs):
        self.graph.add_node(node.left.id_)  # make IR nodes hashable?
        self.visit(node.right, current_write=node.left.id_)
        self.last_write_access[node.left.name] = node.left.id_


def generate_dependency_graph(loops: List[HorizontalLoop]) -> nx.DiGraph:
    return _FieldWriteDependencyGraph().generate(loops)
