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
from eve import Node, NodeTranslator, NodeVisitor
from gtc_unstructured.irs import nir
from gtc_unstructured.irs.nir_passes.field_dependency_graph import generate_dependency_graph


class _FindMergeCandidatesAnalysis(NodeVisitor):
    """Find horizontal loop merge candidates.

    Result is a List[List[HorizontalLoop]], where the inner list contains mergable loops.
    Currently the merge sets are ordered and disjunct, see question below.

    In the following examples A, B, C, ... are loops

    TODO Question
    Should we report all possible merge candidates, example: A, B, C
    - A + B and B + C possible, but not A + B + C  -> want both candidates (currently we only return [A,B])
    - A + B + C possible, we only want A + B + C, but not A + B and B + C as candidates

    Candidates are selected as follows:
    - Different location types cannot be fused

    - Only adjacent loops are considered
      Example: if A, C can be fused but an independent B which cannot be fused (e.g. different location) is in the middle,
      we don't consider A + C for fusion

    - Read after write access
      - if the read is without offset, we can fuse
      - if the read is with offset, we cannot fuse
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.candidates = []
        self.candidate = []

    @classmethod
    def find(cls, root, **kwargs) -> List[List[nir.HorizontalLoop]]:
        """Run the visitor, returns merge candidates."""
        instance = cls()
        instance.visit(root, **kwargs)
        if len(instance.candidate) > 1:
            instance.candidates.append(instance.candidate)
        return instance.candidates

    def has_read_with_offset_after_write(self, graph: nx.DiGraph, **kwargs):
        return any(edge["extent"] for _, _, edge in graph.edges(data=True))

    def visit_HorizontalLoop(self, node: nir.HorizontalLoop, **kwargs):
        if len(self.candidate) == 0:
            self.candidate.append(node)
            return
        elif (
            self.candidate[-1].location_type == node.location_type
        ):  # same location type as previous
            dependencies = generate_dependency_graph(self.candidate + [node])
            if not self.has_read_with_offset_after_write(dependencies):
                self.candidate.append(node)
                return
        # cannot merge to previous loop:
        if len(self.candidate) > 1:
            self.candidates.append(self.candidate)  # add a new merge set
        self.candidate = [node]


def _find_merge_candidates(root: nir.VerticalLoop):
    return _FindMergeCandidatesAnalysis().find(root)


class MergeHorizontalLoops(NodeTranslator):
    @classmethod
    def apply(cls, root: nir.VerticalLoop, merge_candidates, **kwargs) -> nir.VerticalLoop:
        return cls().visit(root, merge_candidates=merge_candidates)

    def visit_VerticalLoop(
        self, node: nir.VerticalLoop, *, merge_candidates: List[List[nir.HorizontalLoop]], **kwargs
    ):
        for candidate in merge_candidates:
            declarations = []
            statements = []
            location_type = candidate[0].location_type

            first_index = node.horizontal_loops.index(candidate[0])
            last_index = node.horizontal_loops.index(candidate[-1])

            for loop in candidate:
                declarations += loop.stmt.declarations
                statements += loop.stmt.statements

            node.horizontal_loops[first_index : last_index + 1] = [  # noqa: E203
                nir.HorizontalLoop(
                    stmt=nir.BlockStmt(
                        declarations=declarations,
                        statements=statements,
                        location_type=location_type,
                    ),
                    location_type=location_type,
                )
            ]

        return node


def merge_horizontal_loops(
    root: nir.VerticalLoop, merge_candidates: List[List[nir.HorizontalLoop]]
):
    return MergeHorizontalLoops().apply(root, merge_candidates)


def find_and_merge_horizontal_loops(root: Node):
    copy = root.copy(deep=True)
    vertical_loops = eve.iter_tree(copy).if_isinstance(nir.VerticalLoop).to_list()
    for loop in vertical_loops:
        assert isinstance(loop, nir.VerticalLoop)
        loop = merge_horizontal_loops(loop, _find_merge_candidates(loop))

    return copy
