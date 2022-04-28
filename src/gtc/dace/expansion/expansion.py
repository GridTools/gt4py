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
from typing import Dict, List, Union

import dace
import dace.data
import dace.library
import dace.subsets
import sympy

import gtc.common as common
import gtc.oir as oir
from gt4py import definitions as gt_def
from gt4py.definitions import Extent
from gtc import daceir as dcir
from gtc.dace.expansion.oir_to_daceir import DaCeIRBuilder
from gtc.dace.expansion.sdfg_builder import StencilComputationSDFGBuilder
from gtc.dace.nodes import StencilComputation


def make_access_subset_dict(
    extent: gt_def.Extent, interval: oir.Interval, axes: List[dcir.Axis]
) -> Dict[dcir.Axis, Union[dcir.IndexWithExtent, dcir.DomainInterval]]:
    from gtc import daceir as dcir

    i_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[0][0], axis=dcir.Axis.I),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[0][1], axis=dcir.Axis.I),
    )
    j_interval = dcir.DomainInterval(
        start=dcir.AxisBound(level=common.LevelMarker.START, offset=extent[1][0], axis=dcir.Axis.J),
        end=dcir.AxisBound(level=common.LevelMarker.END, offset=extent[1][1], axis=dcir.Axis.J),
    )
    k_interval: Union[dcir.IndexWithExtent, dcir.DomainInterval]
    if isinstance(interval, dcir.IndexWithExtent):
        k_interval = interval
    else:
        k_interval = dcir.DomainInterval(
            start=dcir.AxisBound(
                level=interval.start.level,
                offset=interval.start.offset,
                axis=dcir.Axis.K,
            ),
            end=dcir.AxisBound(
                level=interval.end.level, offset=interval.end.offset, axis=dcir.Axis.K
            ),
        )
    res = {dcir.Axis.I: i_interval, dcir.Axis.J: j_interval, dcir.Axis.K: k_interval}
    return {axis: res[axis] for axis in axes}


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
    def fix_context(nsdfg, node: "StencilComputation", parent_state, daceir):

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
    def expansion(
        node: "StencilComputation", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
        start, end = (
            node.oir_node.sections[0].interval.start,
            node.oir_node.sections[0].interval.end,
        )
        for section in node.oir_node.sections:
            start = min(start, section.interval.start)
            end = max(end, section.interval.end)

        overall_interval = dcir.DomainInterval(
            start=dcir.AxisBound(axis=dcir.Axis.K, level=start.level, offset=start.offset),
            end=dcir.AxisBound(axis=dcir.Axis.K, level=end.level, offset=end.offset),
        )
        overall_extent = Extent.zeros(2)
        for he in node.oir_node.iter_tree().if_isinstance(oir.HorizontalExecution):
            overall_extent = overall_extent.union(node.get_extents(he))

        parent_arrays = dict()
        for edge in parent_state.in_edges(node):
            if edge.dst_conn is not None:
                parent_arrays[edge.dst_conn[len("__in_") :]] = parent_sdfg.arrays[edge.data.data]
        for edge in parent_state.out_edges(node):
            if edge.src_conn is not None:
                parent_arrays[edge.src_conn[len("__out_") :]] = parent_sdfg.arrays[edge.data.data]

        daceir_builder_global_ctx = DaCeIRBuilder.GlobalContext(
            library_node=node, block_extents=node.get_extents, arrays=parent_arrays
        )

        iteration_ctx = DaCeIRBuilder.IterationContext.init(
            grid_subset=dcir.GridSubset.from_gt4py_extent(overall_extent).set_interval(
                axis=dcir.Axis.K, interval=overall_interval
            )
        )
        try:
            daceir: dcir.StateMachine = DaCeIRBuilder().visit(
                node.oir_node,
                global_ctx=daceir_builder_global_ctx,
                iteration_ctx=iteration_ctx,
                expansion_specification=list(node.expansion_specification),
            )
        finally:
            iteration_ctx.clear()

        nsdfg = StencilComputationSDFGBuilder().visit(daceir)
        StencilComputationExpansion.fix_context(nsdfg, node, parent_state, daceir)

        return nsdfg
