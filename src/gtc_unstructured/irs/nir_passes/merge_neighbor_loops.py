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

import copy
from typing import Dict, List, cast

from devtools import debug  # noqa: unused

import eve  # noqa: F401
from eve import Node, NodeTranslator
from eve.type_definitions import SymbolName, SymbolRef
from gtc_unstructured.irs import nir


def _find_merge_candidates(h_loop: nir.HorizontalLoop) -> List[List[nir.NeighborLoop]]:
    # This finder is broken and it doesn't compute any data dependency analysis.
    # It will only work for naive cases where neighbor loops are contiguous
    # and they do not need to be reordered.
    merge_groups = []
    neighbor_loops = cast(
        List[nir.NeighborLoop], eve.iter_tree(h_loop).if_isinstance(nir.NeighborLoop).to_list()
    )
    outer = 0
    max_len = len(neighbor_loops)
    while outer < len(neighbor_loops):
        target_connectivity = neighbor_loops[outer].connectivity
        i = outer + 1
        while i < max_len and neighbor_loops[i].connectivity == target_connectivity:
            i += 1
        merge_groups.append(neighbor_loops[outer:i])
        outer = i

    return merge_groups


class RenameSymbol(NodeTranslator):
    @classmethod
    def apply(cls, root: nir.Node, old_name: SymbolName, new_name: SymbolName) -> nir.Node:
        return cls().visit(root, old_name=old_name, new_name=new_name)

    def visit_Node(
        self,
        node: nir.Node,
        *,
        old_name: SymbolName,
        new_name: SymbolName,
        **kwargs,
    ):
        new_fields = {}
        for field_name, field_value in node.iter_children():
            # Due to pydantic and toolchain errors, sometimes SymbolName and SymbolRef
            # are used more or less interchangeably...
            if isinstance(field_value, (SymbolName, SymbolRef)) and field_value == old_name:
                new_fields[field_name] = SymbolRef(new_name)
            else:
                new_fields[field_name] = self.visit(
                    field_value, old_name=old_name, new_name=new_name
                )
        return node.__class__(**new_fields)


class MergeNeighborLoops(NodeTranslator):
    @classmethod
    def apply(
        cls, root: nir.Computation, merge_groups: Dict[str, List[List[nir.NeighborLoop]]], **kwargs
    ) -> nir.Computation:
        return cls().visit(root, merge_groups=merge_groups)

    def visit_HorizontalLoop(
        self,
        node: nir.HorizontalLoop,
        *,
        merge_groups: Dict[str, List[List[nir.NeighborLoop]]],
        **kwargs,
    ):
        assert id(node) in merge_groups
        groups: List[List[nir.NeighborLoop]] = merge_groups[id(node)]

        # the target neighbor loops where groups will be merged
        heads: List[str] = [id(group[0]) for group in groups]

        # mapping from id(NeighborLoop) to its target loop where it should be merged
        # (only for non targets)
        targets: Dict[int, nir.NeighborLoop] = {}

        # mapping from id(NeighborLoop) to the new initialization statements from the
        # merged loops to add in front of the neighbor loop
        targets_init: Dict[int, List[nir.AssignStmt]] = {}

        stmt_declarations = node.stmt.declarations
        stmt_statements = []

        num_stmts = len(node.stmt.statements)
        for i, hl_stmt in enumerate(node.stmt.statements):
            # Traverse all the statements in the horizontal loop
            if isinstance(hl_stmt, nir.NeighborLoop):
                if id(hl_stmt) in heads:
                    # If it is a target neighbor loop, create the dicts
                    # from id(NeighborLoop) to this loop, for all the other
                    # loops that will be merged into this
                    current_group = groups[heads.index(id(hl_stmt))]
                    target_n_loop = copy.deepcopy(current_group[0])
                    assert id(target_n_loop) == id(hl_stmt)
                    for other_n_loop in current_group[1:]:
                        targets[id(other_n_loop)] = target_n_loop
                    stmt_statements.append(target_n_loop)

                else:
                    # If it is a neighbor loop that should be merged,
                    # merge body into target loop
                    assert id(hl_stmt) in targets
                    target_n_loop = targets[id(hl_stmt)]
                    other_body: nir.BlockStmt = RenameSymbol.apply(
                        hl_stmt.body, hl_stmt.name, target_n_loop.name
                    )
                    target_n_loop.body.declarations.extend(other_body.declarations)
                    target_n_loop.body.statements.extend(other_body.statements)

            elif (
                isinstance(hl_stmt, nir.AssignStmt)
                and i < num_stmts - 1
                and isinstance(node.stmt.statements[i + 1], nir.NeighborLoop)
                and id(node.stmt.statements[i + 1]) in targets
            ):
                # If it is the initialization statement of a reduce neighbor loop,
                # save it for later in the list of inits associated to the target loop
                targets_init.setdefault(id(targets[id(node.stmt.statements[i + 1])]), []).append(
                    hl_stmt
                )

            else:
                # Any other statement just passes
                stmt_statements.append(hl_stmt)

        # Move the reduce initialization statements of the merged neighbor loops
        # in front of the target neighbor loop in which they were merged into.
        i = 0
        while i < len(stmt_statements):
            if id(stmt_statements[i]) in targets_init:
                offset = len(targets_init[id(stmt_statements[i])]) + 1
                for init_stmt in targets_init[id(stmt_statements[i])]:
                    stmt_statements.insert(i, init_stmt)
                i += offset
            else:
                i += 1

        return nir.HorizontalLoop(
            iteration_space=node.iteration_space,
            stmt=nir.BlockStmt(declarations=stmt_declarations, statements=stmt_statements),
        )


def find_and_merge_neighbor_loops(root: Node):
    horizontal_loops = eve.iter_tree(root).if_isinstance(nir.HorizontalLoop).to_list()
    merge_groups = {id(h_loop): _find_merge_candidates(h_loop) for h_loop in horizontal_loops}
    new_root = MergeNeighborLoops.apply(root, merge_groups)

    return new_root
