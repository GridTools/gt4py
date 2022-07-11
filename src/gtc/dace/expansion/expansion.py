# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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
import copy
from typing import Dict, List

import dace
import dace.data
import dace.library
import dace.subsets
import sympy

from gtc import daceir as dcir
from gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gtc.dace.expansion.sdfg_builder import StencilComputationSDFGBuilder
from gtc.dace.nodes import StencilComputation

from .utils import split_horizontal_exeuctions_regions


@dace.library.register_expansion(StencilComputation, "default")
class StencilComputationExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def _solve_for_domain(field_decls: Dict[str, dcir.FieldDecl], outer_subsets):
        equations = []
        symbols = set()

        # Collect equations and symbols from arguments and shapes
        for field, decl in field_decls.items():
            inner_shape = [dace.symbolic.pystr_to_symbolic(s) for s in decl.shape]
            outer_shape = [
                dace.symbolic.pystr_to_symbolic(s) for s in outer_subsets[field].bounding_box_size()
            ]

            for inner_dim, outer_dim in zip(inner_shape, outer_shape):
                repldict = {}
                for sym in dace.symbolic.symlist(inner_dim).values():
                    newsym = dace.symbolic.symbol("__SOLVE_" + str(sym))
                    symbols.add(newsym)
                    repldict[sym] = newsym

                # Replace symbols with __SOLVE_ symbols so as to allow
                # the same symbol in the called SDFG
                if repldict:
                    inner_dim = inner_dim.subs(repldict)

                equations.append(inner_dim - outer_dim)
        if len(symbols) == 0:
            return {}

        # Solve for all at once
        results = sympy.solve(equations, *symbols, dict=True)
        result = results[0]
        result = {str(k)[len("__SOLVE_") :]: v for k, v in result.items()}
        return result

    @staticmethod
    def _fix_context(
        nsdfg, node: "StencilComputation", parent_state: dace.SDFGState, daceir: dcir.NestedSDFG
    ):
        """Apply changes to StencilComputation and the SDFG it is embedded in to satisfy post-expansion constraints.

        * change connector names to match inner array name (before expansion prefixed to satisfy uniqueness)
        * change in- and out-edges' subsets so that they have the same shape as the corresponding array inside
        * determine the domain size based on edges to StencilComputation

        """
        # change connector names
        for in_edge in parent_state.in_edges(node):
            assert in_edge.dst_conn.startswith("__in_")
            in_edge.dst_conn = in_edge.dst_conn[len("__in_") :]
        for out_edge in parent_state.out_edges(node):
            assert out_edge.src_conn.startswith("__out_")
            out_edge.src_conn = out_edge.src_conn[len("__out_") :]

        # union input and output subsets
        subsets = {}
        for edge in parent_state.in_edges(node):
            subsets[edge.dst_conn] = edge.data.subset
        for edge in parent_state.out_edges(node):
            subsets[edge.src_conn] = dace.subsets.union(
                edge.data.subset, subsets.get(edge.src_conn, edge.data.subset)
            )
        # ensure single-use of input and output subset instances
        for edge in parent_state.in_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.dst_conn])
        for edge in parent_state.out_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.src_conn])

        # determine "__I", "__J" and "__K" values based on memlets to StencilComputation's shape
        symbol_mapping = StencilComputationExpansion._solve_for_domain(
            {
                decl.name: decl
                for decl in daceir.field_decls
                if decl.name
                in set(memlet.field for memlet in daceir.read_memlets + daceir.write_memlets)
            },
            subsets,
        )
        nsdfg.symbol_mapping.update({**symbol_mapping, **node.symbol_mapping})

    @staticmethod
    def _get_parent_arrays(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> Dict[str, dace.data.Data]:
        parent_arrays: Dict[str, dace.data.Data] = {}
        for edge in (e for e in parent_state.in_edges(node) if e.dst_conn is not None):
            parent_arrays[edge.dst_conn[len("__in_") :]] = parent_sdfg.arrays[edge.data.data]
        for edge in (e for e in parent_state.out_edges(node) if e.src_conn is not None):
            parent_arrays[edge.src_conn[len("__out_") :]] = parent_sdfg.arrays[edge.data.data]
        return parent_arrays

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
        """Expand the coarse SDFG in parent_sdfg to a NestedSDFG with all the states."""
        split_horizontal_exeuctions_regions(node)
        arrays = StencilComputationExpansion._get_parent_arrays(node, parent_state, parent_sdfg)

        daceir: dcir.NestedSDFG = DaCeIRBuilder().visit(
            node.oir_node, global_ctx=DaCeIRBuilder.GlobalContext(library_node=node, arrays=arrays)
        )

        nsdfg = StencilComputationSDFGBuilder().visit(daceir)

        StencilComputationExpansion._fix_context(nsdfg, node, parent_state, daceir)

        return nsdfg
