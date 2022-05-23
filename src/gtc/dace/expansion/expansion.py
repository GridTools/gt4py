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

        for in_edge in parent_state.in_edges(node):
            assert in_edge.dst_conn.startswith("__in_")
            in_edge.dst_conn = in_edge.dst_conn[len("__in_") :]
        for out_edge in parent_state.out_edges(node):
            assert out_edge.src_conn.startswith("__out_")
            out_edge.src_conn = out_edge.src_conn[len("__out_") :]

        subsets = dict()
        for edge in parent_state.in_edges(node):
            subsets[edge.dst_conn] = edge.data.subset
        for edge in parent_state.out_edges(node):
            subsets[edge.src_conn] = dace.subsets.union(
                edge.data.subset, subsets.get(edge.src_conn, edge.data.subset)
            )
        for edge in parent_state.in_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.dst_conn])
        for edge in parent_state.out_edges(node):
            edge.data.subset = copy.deepcopy(subsets[edge.src_conn])

        symbol_mapping = StencilComputationExpansion._solve_for_domain(
            {
                name: decl
                for name, decl in daceir.field_decls.items()
                if name
                in set(memlet.field for memlet in daceir.read_memlets + daceir.write_memlets)
            },
            subsets,
        )
        if "__K" in nsdfg.sdfg.free_symbols and "__K" not in symbol_mapping:
            symbol_mapping["__K"] = 0
        nsdfg.symbol_mapping.update({**symbol_mapping, **node.symbol_mapping})

    @staticmethod
    def _get_parent_arrays(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ):
        parent_arrays = dict()
        for edge in parent_state.in_edges(node):
            if edge.dst_conn is not None:
                parent_arrays[edge.dst_conn[len("__in_") :]] = parent_sdfg.arrays[edge.data.data]
        for edge in parent_state.out_edges(node):
            if edge.src_conn is not None:
                parent_arrays[edge.src_conn[len("__out_") :]] = parent_sdfg.arrays[edge.data.data]
        return parent_arrays

    @staticmethod
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:

        global_ctx = DaCeIRBuilder.GlobalContext(
            library_node=node,
            arrays=StencilComputationExpansion._get_parent_arrays(node, parent_state, parent_sdfg),
        )

        daceir: dcir.NestedSDFG = DaCeIRBuilder().visit(
            node.oir_node,
            global_ctx=global_ctx,
        )

        nsdfg = StencilComputationSDFGBuilder().visit(daceir)

        StencilComputationExpansion._fix_context(nsdfg, node, parent_state, daceir)

        return nsdfg
