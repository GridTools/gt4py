# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
from typing import Dict, List, Sequence, Set, Tuple, Union

import dace
import dace.codegen.control_flow as cf
import dace.sdfg.graph
import dace.symbolic
import sympy
from dace.properties import DictProperty, make_properties
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import transformation
from dace.transformation.helpers import nest_sdfg_subgraph

from gt4py import eve
from gt4py.cartesian.gtc import common, daceir as dcir, oir
from gt4py.cartesian.gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gt4py.cartesian.gtc.dace.expansion_specification import ExpansionItem, Iteration, Map, Skip
from gt4py.cartesian.gtc.dace.nodes import StencilComputation
from gt4py.cartesian.gtc.dace.utils import make_dace_subset, union_inout_memlets
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents


AccessInfoCollection = Dict[eve.SymbolRef, dcir.FieldAccessInfo]
SymblicAxisDict = Dict[dcir.Axis, dace.symbolic.SymbolicType]
SymbolicSplitType = Tuple[SymblicAxisDict, Dict[eve.SymbolRef, SymblicAxisDict]]


#
def _get_field_access_infos(
    state: dace.SDFGState,
    node: StencilComputation,
) -> Tuple[Dict[eve.SymbolName, dcir.FieldAccessInfo], Dict[eve.SymbolName, dcir.FieldAccessInfo]]:
    """Return a dictionary which maps field names to access infos."""
    parent_arrays: Dict[str, dace.data.Data] = {}
    for edge in (e for e in state.in_edges(node) if e.dst_conn is not None):
        parent_arrays[edge.dst_conn[len("__in_") :]] = state.parent.arrays[edge.data.data]
    for edge in (e for e in state.out_edges(node) if e.src_conn is not None):
        parent_arrays[edge.src_conn[len("__out_") :]] = state.parent.arrays[edge.data.data]

    daceir: dcir.NestedSDFG = DaCeIRBuilder().visit(
        node.oir_node,
        global_ctx=DaCeIRBuilder.GlobalContext(library_node=node, arrays=parent_arrays),
    )

    read_memlets, write_memlets, _ = union_inout_memlets(daceir.states)
    return {mem.field: mem.access_info for mem in read_memlets}, {
        mem.field: mem.access_info for mem in write_memlets
    }


def _union_access_infos(
    first: Dict[eve.SymbolRef, dcir.FieldAccessInfo],
    second: Dict[eve.SymbolRef, dcir.FieldAccessInfo],
) -> Dict[eve.SymbolRef, dcir.FieldAccessInfo]:
    res = dict(first)
    for key, value in second.items():
        res[key] = value.union(res.get(key, value))
    return res


def _make_dace_subset_symbolic_context(
    context_info: Tuple[SymblicAxisDict, SymblicAxisDict],
    access_info: dcir.FieldAccessInfo,
    old_range: dace.subsets.Range,
    data_dims: Tuple[int, ...],
    tile_sizes,
) -> dace.subsets.Range:
    context_domain, context_origin = context_info
    clamped_access_info = access_info
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes:
            if axis in tile_sizes:
                context_origin[axis] = max(context_origin[axis], 0)
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
    res_ranges = []

    for ax_idx, axis in enumerate(clamped_access_info.axes()):
        if axis in context_domain:
            clamped_interval = clamped_access_info.grid_subset.intervals[axis]
            if isinstance(clamped_interval, dcir.DomainInterval):
                subset_start, subset_end = clamped_interval.start, clamped_interval.end
                subset_start_sym = context_origin[axis] + subset_start.offset
                subset_end_sym = context_origin[axis] + subset_end.offset
                if subset_start.level == common.LevelMarker.END:
                    subset_start_sym += context_domain[axis]
                if subset_end.level == common.LevelMarker.END:
                    subset_end_sym += context_domain[axis]
            else:
                assert isinstance(clamped_interval, dcir.TileInterval)
                subset_start_offset, subset_end_offset = (
                    clamped_interval.start_offset,
                    clamped_interval.end_offset,
                )
                subset_start_sym = (
                    context_origin[axis] + subset_start_offset + axis.tile_dace_symbol()
                )
                subset_end_sym = (
                    context_origin[axis]
                    + subset_end_offset
                    + axis.tile_dace_symbol()
                    + tile_sizes[axis]
                )
            res_ranges.append((subset_start_sym, subset_end_sym - 1, 1))
        else:
            res_ranges.append(old_range.ranges[ax_idx])
    res_ranges.extend((0, dim - 1, 1) for dim in data_dims)
    return dace.subsets.Range(res_ranges)


def _nest_sdfg_subgraph(sdfg: dace.SDFG, subgraph):
    if len(subgraph.nodes()) == 1:
        # nest_sdfg_subgraph does not apply on single-state subgraphs, so we add an empty state before to trigger
        # nesting.
        new_state = sdfg.add_state_before(subgraph.source_nodes()[0])
        subgraph = dace.sdfg.graph.SubgraphView(sdfg, [new_state, *subgraph.nodes()])

    for edge in (e for n in subgraph.nodes() for e in subgraph.in_edges(n)):
        for sym in edge.data.assignments.keys():
            if sym not in sdfg.symbols:
                sdfg.add_symbol(sym, stype=dace.float64)
    nsdfg_state: dace.SDFGState = nest_sdfg_subgraph(sdfg, subgraph)
    if nsdfg_state.label == "symbolic_output":
        nsdfg_state = next(iter(nsdfg_state.parent.predecessor_states(nsdfg_state)))

    nsdfg_node: dace.nodes.NestedSDFG = next(
        iter(node for node in nsdfg_state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    )

    nsdfg_node.sdfg.parent_sdfg = sdfg
    nsdfg_node.sdfg.parent_nsdfg_node = nsdfg_node
    for n, s in nsdfg_node.sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.NestedSDFG):
            n.sdfg.parent_sdfg = s.parent
            n.sdfg.parent_nsdfg_node = n

    return nsdfg_state, nsdfg_node


@make_properties
class PartialExpansion(transformation.SubgraphTransformation):
    _inconsistent_split_msg = "inconsistent symbolic split"
    strides = DictProperty(
        key_type=dcir.Axis,
        value_type=int,
        default={dcir.Axis.I: 8, dcir.Axis.J: 8},
        desc="Map dimensions to partially expand to the corresponding map stride.",
    )

    class _InconsistentSymbolicSplit(Exception):
        pass

    @staticmethod
    def subgraph_all_nodes_recursive_topological(
        subgraph: Union[dace.sdfg.graph.SubgraphView, dace.SDFG]
    ):
        for state in dfs_topological_sort(subgraph):
            for node in dfs_topological_sort(state):
                if isinstance(node, dace.nodes.NestedSDFG):
                    yield from PartialExpansion.subgraph_all_nodes_recursive_topological(node.sdfg)
                yield node, state
            yield state, state.parent

    @staticmethod
    def _nsdfg_checks(sdfg: dace.SDFG):
        cfg = cf.structured_control_flow_tree(sdfg, lambda _: "")
        if not all(isinstance(c, cf.SingleState) for c in cfg.children):
            return False
        return True

    def _test_compatible_regions(self, stencil_computations):
        for node, _ in stencil_computations:
            for region in node.oir_node.walk_values().if_isinstance(oir.HorizontalRestriction):
                if dcir.Axis.I in self.strides:
                    if region.mask.i.start is not None or region.mask.i.end is not None:
                        return False
                if dcir.Axis.J in self.strides:
                    if region.mask.j.start is not None or region.mask.j.end is not None:
                        return False
        return True

    def _make_new_expansion_specification(
        self, node: StencilComputation, dims: Sequence[str]
    ) -> List[ExpansionItem]:
        current_expansion_specification = list(node.expansion_specification)
        res_expansion_specification = [
            Map(
                iterations=[
                    Iteration(axis=dcir.Axis(dim), kind="tiling", stride=self.strides[dim])
                    for dim in dims
                ]
            )
        ]
        for item_idx, item in enumerate(list(current_expansion_specification)):
            iterations: List[Iteration]
            if isinstance(item, Map):
                iterations = list(item.iterations)
            else:
                res_expansion_specification.append(copy.deepcopy(item))
                continue

            for it_idx, it in reversed(list(enumerate(iterations))):
                if str(it.axis) in dims and it.kind == "tiling":
                    del iterations[it_idx]

            if iterations:
                res_expansion_specification.append(
                    Map(iterations=[copy.deepcopy(it) for it in iterations])
                )

        return res_expansion_specification

    def _move_dim_outermost(self, node: StencilComputation, dims: Sequence[str]):
        node.expansion_specification = self._make_new_expansion_specification(node, dims)

    def _test_dim_outermost(self, node: StencilComputation, dims: Sequence[str]):
        from gt4py.cartesian.gtc.dace.nodes import make_expansion_order

        new_spec = self._make_new_expansion_specification(node, dims)
        try:
            make_expansion_order(node, new_spec)
        except ValueError:
            return False
        return True

    def can_be_applied(self, sdfg: dace.SDFG, subgraph: dace.sdfg.graph.SubgraphView) -> bool:
        stencil_computations = list(
            filter(
                lambda n: isinstance(n[0], StencilComputation),
                PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph),
            )
        )

        # test if any computations in the graph
        if not subgraph.nodes():
            return False

        if not stencil_computations:
            return False

        for nsdfg, _ in filter(
            lambda n: isinstance(n[0], dace.nodes.NestedSDFG),
            PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph),
        ):
            if not PartialExpansion._nsdfg_checks(nsdfg.sdfg):
                return False

        # E.g. Maps in subgraphs are not supported
        if not all(
            isinstance(
                n,
                (dace.nodes.AccessNode, dace.SDFGState, StencilComputation, dace.nodes.NestedSDFG),
            )
            for n, _ in PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph)
        ):
            return False

        # We can't infer the semantic correspondance of subsets on edges between access nodes, there fore we can't tile
        # those and they have to be excluded
        if any(
            isinstance(edge.src, dace.nodes.AccessNode)
            and isinstance(edge.dst, dace.nodes.AccessNode)
            for state in subgraph.nodes()
            for edge in state.edges()
        ):
            return False

        # test if moving the partial expanded dimensions outermost is supported
        for node, _ in stencil_computations:
            if not self._test_dim_outermost(node, list(self.strides)):
                return False

        if not self._test_compatible_regions(stencil_computations):
            return False

        if not self._test_symbolic_split(sdfg, subgraph):
            return False

        if not self._test_extents_compatible(stencil_computations):
            return False

        return True

    def _test_symbolic_split(
        self,
        sdfg: dace.SDFG,
        subgraph: Union[dace.SDFG, dace.sdfg.graph.SubgraphView],
    ) -> bool:
        try:
            self._get_symbolic_splits(sdfg, subgraph)
        except PartialExpansion._InconsistentSymbolicSplit:
            return False
        return True

    class _Renamer(eve.NodeTranslator):
        def __init__(self, name_map):
            self.name_map = name_map

        def visit_VerticalLoop(self, node: oir.VerticalLoop):
            return oir.VerticalLoop(
                loop_order=node.loop_order, sections=self.visit(node.sections), caches=[]
            )

        def visit_FieldAccess(self, node: oir.FieldAccess):
            offset = self.visit(node.offset)
            data_index = self.visit(node.data_index)
            return oir.FieldAccess(
                name=self.name_map[node.name],
                offset=offset,
                data_index=data_index,
                dtype=node.dtype,
            )

    def _test_extents_compatible(
        self, nodes: List[Tuple[StencilComputation, dace.SDFGState]]
    ) -> bool:
        vertical_loops = []
        temporary_decls = {}
        params: Dict[eve.SymbolRef, Union[oir.FieldDecl, oir.ScalarDecl]] = {}
        for node, state in nodes:
            name_map = PartialExpansion._get_name_map(state, node)
            vl = PartialExpansion._Renamer(name_map).visit(node.oir_node)
            vertical_loops.append(vl)

            for name, decl in node.declarations.items():
                if isinstance(decl, oir.Temporary):
                    temporary_decls[name_map[name]] = oir.Temporary(
                        name=name_map[name],
                        dtype=decl.dtype,
                        dimensions=decl.dimensions,
                        data_dims=decl.data_dims,
                    )
                elif isinstance(decl, oir.FieldDecl):
                    params[name_map[name]] = oir.FieldDecl(
                        name=name_map[name],
                        dtype=decl.dtype,
                        dimensions=decl.dimensions,
                        data_dims=decl.data_dims,
                    )
                else:
                    assert isinstance(decl, oir.ScalarDecl)
                    params[name] = decl
        oir_stencil = oir.Stencil(
            name="__adhoc",
            params=list(params.values()),
            vertical_loops=vertical_loops,
            declarations=list(temporary_decls.values()),
        )

        oir_extents = compute_horizontal_block_extents(oir_stencil)

        for i, (node, _) in enumerate(nodes):
            for j, section in enumerate(node.oir_node.sections):
                for k, he in enumerate(section.horizontal_executions):
                    he_id = id(vertical_loops[i].sections[j].horizontal_executions[k])
                    for ax_idx, axis in enumerate(dcir.Axis.dims_horizontal()):
                        if (
                            axis in self.strides
                            and not oir_extents[he_id][ax_idx] == node.get_extents(he)[ax_idx]
                        ):
                            return False
        return True

    def _solve_domain_subsets(self, sdfg, node, access_infos, outer_subsets):
        # Collect equations and symbols from arguments and shapes
        equations = []
        symbols = set()

        for field, info in access_infos.items():
            if field not in outer_subsets:
                assert isinstance(node, dace.nodes.NestedSDFG)
                # field is transient in nested sdfg
                continue
            for axis in info.axes():
                if axis in info.variable_offset_axes:
                    info = info.clamp_full_axis(axis)
            inner_exprs = [dace.symbolic.pystr_to_symbolic(s) for s in info.shape]
            outer_exprs = [
                dace.symbolic.pystr_to_symbolic(s) for s in outer_subsets[field].bounding_box_size()
            ]
            # strip data dims
            outer_exprs = outer_exprs[: len(inner_exprs)]

            for ax_idx, axis in enumerate(info.axes()):
                if axis not in self.strides:
                    continue
                inner_expr = inner_exprs[ax_idx]
                outer_expr = outer_exprs[ax_idx]
                repldict = {}
                for sym in dace.symbolic.symlist(inner_expr).values():
                    newsym = dace.symbolic.symbol("__SOLVE_" + str(sym))
                    symbols.add(newsym)
                    repldict[sym] = newsym
                # Replace symbols with __SOLVE_ symbols so as to allow
                # the same symbol in the called SDFG
                if repldict:
                    inner_expr = inner_expr.subs(repldict)

                equations.append(inner_expr - outer_expr)
        # Solve for all at once
        results = sympy.solve(equations, *symbols, dict=True)
        if not results:
            raise PartialExpansion._InconsistentSymbolicSplit("inconsistent symbolic split.")
        result = results[0]
        result = {str(k)[len("__SOLVE_") :]: v for k, v in result.items()}
        return {
            axis: result[axis.domain_symbol()]
            for axis in self.strides.keys()
            if axis.domain_symbol() in result
        }

    def _get_symbolic_origin_and_domain(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        node: Union[dace.nodes.NestedSDFG, StencilComputation],
        access_infos: Dict[eve.SymbolRef, dcir.FieldAccessInfo],
    ) -> SymbolicSplitType:
        """Determine which part of the memlet subsets corresponds to halo and domain, respectively.

        This is done by matching the `node`'s in/out memlet subsets with the dcir.GridSubset in `access_infos`.

        Returns a dict mapping the field names to a 2-tuple with
        * the origin of the field in the node as absolute index in the parent sdfg's array
        * the shape of the domain as tuple of symbolic values
        """
        # union input and output subsets
        outer_subsets = {}
        for edge in state.in_edges(node):
            outer_subsets[edge.data.data] = edge.data.subset
        for edge in state.out_edges(node):
            outer_subsets[edge.data.data] = dace.subsets.union(
                edge.data.subset, outer_subsets.get(edge.data.data, edge.data.subset)
            )
        # ensure single-use of input and output subset instances
        for edge in state.in_edges(node):
            edge.data.subset = copy.deepcopy(outer_subsets[edge.data.data])
        for edge in state.out_edges(node):
            edge.data.subset = copy.deepcopy(outer_subsets[edge.data.data])

        origins: Dict[eve.SymbolRef, SymblicAxisDict] = dict()
        solved_domain = self._solve_domain_subsets(sdfg, node, access_infos, outer_subsets)
        symbol_mapping_domain: SymblicAxisDict = {
            ax: node.symbol_mapping[ax.domain_symbol()]
            for ax in self.strides.keys()
            if ax.domain_symbol() in node.symbol_mapping
        }
        domain = self._check_and_union_spatial_tuple(solved_domain, symbol_mapping_domain)

        for field, info in access_infos.items():
            if field not in outer_subsets:
                assert isinstance(node, dace.nodes.NestedSDFG)
                # field is transient in nested sdfg
                continue
            outer_subset: dace.subsets.Range = outer_subsets[field]
            filtered_outer_ranges = {
                ax: orig for ax, orig in zip(info.axes(), outer_subset.ranges) if ax in self.strides
            }
            inner_origin = {
                axis: info.grid_subset.intervals[axis].to_dace_symbolic()[0]
                for axis in self.strides.keys()
                if axis in info.grid_subset.intervals
            }
            origin = {
                axis: dace.symbolic.pystr_to_symbolic(fo[0] - inner_origin[axis])
                for axis, fo in filtered_outer_ranges.items()
            }
            for axis, orig in origin.items():
                origin[axis] = orig.subs(
                    {ax.domain_dace_symbol(): value for ax, value in domain.items()},
                    simultaneous=True,
                )
            origins[field] = origin
            for ax_idx, axis in enumerate(info.axes()):
                if axis in self.strides:
                    interval = info.grid_subset.intervals[axis]
                    assert isinstance(interval, dcir.DomainInterval)
                    interval_end = interval.end
                    if interval_end.level == common.LevelMarker.END:
                        domain = self._check_and_union_spatial_tuple(
                            domain,
                            {
                                axis: (outer_subset.ranges[ax_idx][1] + 1 - interval_end.offset)
                                - origins[field][axis]
                            },
                        )
        return domain, origins

    @staticmethod
    def _check_and_union_spatial_tuple(
        first: SymblicAxisDict, second: SymblicAxisDict
    ) -> SymblicAxisDict:
        res = copy.deepcopy(first)
        for key, value in second.items():
            # comparison with True required due to how sympy works
            if key in res and not (res[key] == value) == True:  # noqa: E712 comparison to True
                raise PartialExpansion._InconsistentSymbolicSplit("inconsistent symbolic split.")
            res[key] = value
        return res

    @staticmethod
    def _union_splits(
        first: Dict[int, SymbolicSplitType], second: Dict[int, SymbolicSplitType]
    ) -> Dict[int, SymbolicSplitType]:
        global_res = copy.deepcopy(first)
        for sdfg_id, (second_domain, second_origins) in second.items():
            sdfg_res_domain, sdfg_res_origins = copy.deepcopy(
                global_res.get(sdfg_id, (second_domain, second_origins))
            )
            sdfg_res_domain = PartialExpansion._check_and_union_spatial_tuple(
                sdfg_res_domain, second_domain
            )
            for field, origin in second_origins.items():
                sdfg_res_origins[field] = PartialExpansion._check_and_union_spatial_tuple(
                    sdfg_res_origins.get(field, origin), origin
                )
            global_res[sdfg_id] = sdfg_res_domain, sdfg_res_origins
        return global_res

    def _get_symbolic_splits(
        self, sdfg: dace.SDFG, subgraph: Union[dace.SDFG, dace.sdfg.graph.SubgraphView]
    ) -> Tuple[AccessInfoCollection, Dict[int, SymbolicSplitType]]:
        splits: Dict[int, SymbolicSplitType] = dict()
        subgraph_access_infos: Dict[eve.SymbolRef, dcir.FieldAccessInfo] = dict()
        for state in subgraph.nodes():
            assert state.parent is sdfg
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    node_access_infos, nsdfg_splits = self._get_symbolic_splits(
                        node.sdfg, node.sdfg
                    )
                    splits = PartialExpansion._union_splits(splits, nsdfg_splits)
                elif isinstance(node, StencilComputation):
                    node_access_infos = node.access_infos
                else:
                    continue
                node_access_infos = PartialExpansion._rename_dict_keys(
                    PartialExpansion._get_name_map(state, node), node_access_infos
                )
                node_split = self._get_symbolic_origin_and_domain(
                    sdfg, state, node, node_access_infos
                )
                splits = PartialExpansion._union_splits(splits, {sdfg.sdfg_id: node_split})
                subgraph_access_infos = _union_access_infos(
                    subgraph_access_infos, node_access_infos
                )

        if isinstance(subgraph, dace.sdfg.graph.SubgraphView):
            return subgraph_access_infos, splits

        return self._get_array_access_info(sdfg, subgraph_access_infos, splits), splits

    @staticmethod
    def _get_name_map(
        state: dace.SDFGState, node: Union[StencilComputation, dace.nodes.NestedSDFG]
    ):
        if isinstance(node, StencilComputation):

            def in_conn_to_name(conn):
                return conn[len("__in_") :]

            def out_conn_to_name(conn):
                return conn[len("__out_") :]

        else:

            def in_conn_to_name(conn):
                return conn

            def out_conn_to_name(conn):
                return conn

        res = dict()
        for edge in filter(lambda e: not e.data.is_empty(), state.in_edges(node)):
            inner_name = in_conn_to_name(edge.dst_conn)
            outer_name = edge.data.data
            res[inner_name] = outer_name
        for edge in filter(lambda e: not e.data.is_empty(), state.out_edges(node)):
            inner_name = out_conn_to_name(edge.src_conn)
            outer_name = edge.data.data
            res[inner_name] = outer_name
        return res

    @staticmethod
    def _rename_dict_keys(name_map, input_dict):
        return {name_map.get(key, key): value for key, value in input_dict.items()}

    @staticmethod
    def _update_shapes(
        sdfg: dace.SDFG, access_infos: Dict[eve.SymbolName, dcir.FieldAccessInfo], domain_dict
    ):
        for name, access_info in access_infos.items():
            for axis in access_info.axes():
                if axis in access_info.variable_offset_axes:
                    access_info = access_info.clamp_full_axis(axis)
            array = sdfg.arrays[name]
            if array.transient:
                shape = [
                    new_shape if ax in domain_dict else old_shape
                    for ax, new_shape, old_shape in zip(
                        access_info.axes(), access_info.overapproximated_shape, array.shape
                    )
                ]
                ndim = len(shape)
                shape += list(array.shape[ndim:])
                shape = [dace.symbolic.pystr_to_symbolic(s) for s in shape]
                for i, s in enumerate(shape):
                    shape[i] = s.subs(
                        {ax.domain_dace_symbol(): value for ax, value in domain_dict.items()},
                        simultaneous=True,
                    )
                strides = [1]
                total_size = shape[0]
                for s in reversed(shape[1:]):
                    strides = [s * strides[0], *strides]
                    total_size *= s
                array.shape = shape
                array.strides = strides
                array.total_size = total_size
                array.storage = dace.StorageType.CPU_ThreadLocal
                array.lifetime = dace.AllocationLifetime.Persistent
                sdfg.parent_nsdfg_node.no_inline = True
            else:
                shape = [
                    new_shape if ax in domain_dict else old_shape
                    for ax, new_shape, old_shape in zip(
                        access_info.axes(), access_info.overapproximated_shape, array.shape
                    )
                ]
                shape = [dace.symbolic.pystr_to_symbolic(s) for s in shape]
                for i, s in enumerate(shape):
                    shape[i] = s.subs(
                        {ax.domain_dace_symbol(): value for ax, value in domain_dict.items()},
                        simultaneous=True,
                    )
                ndim = len(shape)
                shape += list(array.shape[ndim:])
                array.shape = shape

    def _get_array_access_info(self, sdfg, access_infos, symbolic_splits):
        sdfg_access_infos = dict()
        split_domain = symbolic_splits[sdfg.sdfg_id][0]
        for field, info in access_infos.items():
            if field not in sdfg.arrays:
                continue
            array = sdfg.arrays[field]
            split_origin = symbolic_splits[sdfg.sdfg_id][1][field]
            res_intervals = dict(info.grid_subset.intervals)
            for ax_idx, axis in enumerate(info.axes()):
                if axis in self.strides:
                    interval = info.grid_subset.intervals[axis]
                    assert isinstance(interval, dcir.DomainInterval)
                    if interval.start.level == common.LevelMarker.START:
                        start = dcir.AxisBound(
                            axis=axis,
                            level=common.LevelMarker.START,
                            offset=-int(split_origin[axis]),
                        )
                    else:
                        raise AssertionError
                    if interval.end.level == common.LevelMarker.END:
                        offset = array.shape[ax_idx] - split_domain[axis] - split_origin[axis]
                        end = dcir.AxisBound(
                            axis=axis, level=common.LevelMarker.END, offset=int(offset)
                        )
                    else:
                        raise AssertionError
                    res_intervals[axis] = dcir.DomainInterval(start=start, end=end)
            res_grid_subset = dcir.GridSubset(intervals=res_intervals)
            sdfg_access_infos[field] = dcir.FieldAccessInfo(
                grid_subset=res_grid_subset,
                global_grid_subset=info.global_grid_subset,
                dynamic_access=info.dynamic_access,
                variable_offset_axes=info.variable_offset_axes,
            )
        return sdfg_access_infos

    def _update_nested_and_gather_access_info(self, nsdfg_node, symbolic_splits, axes):
        all_access_infos = dict()
        nsdfg_access_infos = dict()
        for state in nsdfg_node.sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    node_access_infos = self._update_nested_and_gather_access_info(
                        node, symbolic_splits, axes
                    )
                    name_map = dict()
                    for edge in state.in_edges(node):
                        name_map[edge.dst_conn] = edge.data.data
                    for edge in state.out_edges(node):
                        name_map[edge.src_conn] = edge.data.data
                    node_access_infos = {
                        k: v for k, v in node_access_infos.items() if k in name_map
                    }
                    node_access_infos = PartialExpansion._rename_dict_keys(
                        name_map, node_access_infos
                    )
                    all_access_infos[node] = node_access_infos, node_access_infos, node_access_infos
                elif isinstance(node, StencilComputation):
                    to_sdfg_name_map = PartialExpansion._get_name_map(state, node)
                    in_node_access_infos, out_node_access_infos = _get_field_access_infos(
                        state, node
                    )
                    in_node_access_infos = PartialExpansion._rename_dict_keys(
                        to_sdfg_name_map, in_node_access_infos
                    )
                    out_node_access_infos = PartialExpansion._rename_dict_keys(
                        to_sdfg_name_map, out_node_access_infos
                    )
                    node_access_infos = _union_access_infos(
                        in_node_access_infos, out_node_access_infos
                    )
                    all_access_infos[node] = (
                        node_access_infos,
                        in_node_access_infos,
                        out_node_access_infos,
                    )
                else:
                    continue
                nsdfg_access_infos = _union_access_infos(nsdfg_access_infos, node_access_infos)

        # update shapes and subsets of new nested SDFG
        domain = symbolic_splits[nsdfg_node.sdfg.sdfg_id][0]
        PartialExpansion._update_shapes(nsdfg_node.sdfg, nsdfg_access_infos, domain)

        for state in nsdfg_node.sdfg.states():
            for node in filter(
                lambda n: isinstance(n, (dace.nodes.NestedSDFG, StencilComputation)), state.nodes()
            ):
                node_access_infos, in_access_infos, out_access_infos = all_access_infos[node]
                if isinstance(node, StencilComputation):
                    node.access_infos = PartialExpansion._rename_dict_keys(
                        {v: k for k, v in PartialExpansion._get_name_map(state, node).items()},
                        node_access_infos,
                    )
                for edge in state.in_edges(node) + state.out_edges(node):
                    access_infos = (
                        in_access_infos if edge in state.in_edges(node) else out_access_infos
                    )
                    name = edge.data.data
                    access_info = access_infos[name]
                    naxes = len(access_info.grid_subset.intervals)
                    data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                    new_subset = make_dace_subset(
                        nsdfg_access_infos[name], access_info, data_dims=data_dims
                    )
                    new_subset.replace({ax.domain_dace_symbol(): v for ax, v in domain.items()})
                    edge.data.subset.ranges = list(
                        new_rng if ax in domain else old_rng
                        for ax, new_rng, old_rng in zip(
                            nsdfg_access_infos[name].axes(),
                            new_subset.ranges,
                            edge.data.subset.ranges,
                        )
                    ) + list((0, d - 1, 1) for d in data_dims)

        nsdfg_node.symbol_mapping.update({ax.tile_symbol(): ax.tile_dace_symbol() for ax in axes})
        for ax in axes:
            if ax.tile_symbol() not in nsdfg_node.sdfg.symbols:
                nsdfg_node.sdfg.add_symbol(ax.tile_symbol(), stype=dace.int32)

        return nsdfg_access_infos

    def _set_skips(self, state: dace.SDFGState, node: StencilComputation):
        self._move_dim_outermost(node, list(self.strides.keys()))
        expansion_specification = list(node.expansion_specification)
        item = expansion_specification[0]
        assert isinstance(item, Map)
        if all(it.axis in self.strides for it in item.iterations):
            expansion_specification[0] = Skip(item=item)
        else:
            skip_iterations = [it for it in item.iterations if it.axis in self.strides]
            noskip_iterations = [it for it in item.iterations if it.axis not in self.strides]
            expansion_specification = [
                Skip(item=Map(iterations=skip_iterations, schedule=item.schedule)),
                Map(iterations=noskip_iterations, schedule=dace.ScheduleType.Default),
                *expansion_specification[1:],
            ]
        node.expansion_specification = expansion_specification

        parent_arrays: Dict[str, dace.data.Data] = {}
        for edge in (e for e in state.in_edges(node) if e.dst_conn is not None):
            parent_arrays[edge.dst_conn[len("__in_") :]] = state.parent.arrays[edge.data.data]
        for edge in (e for e in state.out_edges(node) if e.src_conn is not None):
            parent_arrays[edge.src_conn[len("__out_") :]] = state.parent.arrays[edge.data.data]

        daceir: dcir.NestedSDFG = DaCeIRBuilder().visit(
            node.oir_node,
            global_ctx=DaCeIRBuilder.GlobalContext(library_node=node, arrays=parent_arrays),
        )

        _, _, memlets = union_inout_memlets(daceir.states)
        node.access_infos = {mem.field: mem.access_info for mem in memlets}

    def apply(self, sdfg: dace.SDFG):
        # We build a subgraph that the tile map will surround
        subgraph = self.subgraph_view(sdfg)
        nsdfg_state, nsdfg_node = _nest_sdfg_subgraph(sdfg, subgraph)

        stencil_computations: List[Tuple[StencilComputation, dace.SDFGState]] = list(
            filter(
                lambda n: isinstance(n[0], StencilComputation),
                PartialExpansion.subgraph_all_nodes_recursive_topological(nsdfg_node.sdfg),
            )
        )
        axes = set(self.strides.keys())

        _, parent_symbolic_splits = self._get_symbolic_splits(
            sdfg, dace.sdfg.graph.SubgraphView(sdfg, [nsdfg_state])
        )
        domain_dict = parent_symbolic_splits[sdfg.sdfg_id][0]

        # Tag partial expanded dims so they are treated as such
        # during expansion
        for node, state in stencil_computations:
            self._set_skips(state, node)

        # All schedules are expected to be the same - we grab the first stencil
        first_item = stencil_computations[0][0].expansion_specification[0]
        assert isinstance(first_item, Skip) and isinstance(first_item.item, Map)
        tiling_schedule = first_item.item.schedule

        # Collect all nodes and states access info recursively
        # and update the tiling of edges inside the nested SDFG
        nsdfg_access_infos = self._update_nested_and_gather_access_info(
            nsdfg_node, parent_symbolic_splits, axes
        )

        # Reconnect the new subgraph SDFG to the top tiling map
        tiling_ranges = {
            axis.tile_symbol(): str(
                dace.subsets.Range([(0, domain_dict[axis] - 1, self.strides[axis])])
            )
            for axis in axes
        }
        map_entry, map_exit = nsdfg_state.add_map(
            stencil_computations[0][0].label + "_tiling_map",
            ndrange=tiling_ranges,
            schedule=tiling_schedule,
            debuginfo=nsdfg_node.debuginfo,
        )
        for edge in nsdfg_state.in_edges(nsdfg_node):
            memlet = edge.data
            name = memlet.data
            if name in nsdfg_access_infos:
                access_info = nsdfg_access_infos[name]
                naxes = len(access_info.grid_subset.intervals)
                data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                new_subset = _make_dace_subset_symbolic_context(
                    (
                        parent_symbolic_splits[sdfg.sdfg_id][0],
                        parent_symbolic_splits[sdfg.sdfg_id][1][name],
                    ),
                    access_info,
                    memlet.subset,
                    data_dims=data_dims,
                    tile_sizes=self.strides,
                )
            else:
                new_subset = edge.data.subset
            map_entry.add_in_connector("IN_" + name)
            map_entry.add_out_connector("OUT_" + name)
            nsdfg_state.add_edge(edge.src, edge.src_conn, map_entry, "IN_" + name, memlet=edge.data)
            nsdfg_state.add_edge(
                map_entry,
                "OUT_" + name,
                nsdfg_node,
                edge.dst_conn,
                memlet=dace.Memlet(data=name, subset=new_subset),
            )
            nsdfg_state.remove_edge(edge)
        for edge in nsdfg_state.out_edges(nsdfg_node):
            memlet = edge.data
            name = memlet.data
            if name in nsdfg_access_infos:
                access_info = nsdfg_access_infos[name]
                naxes = len(access_info.grid_subset.intervals)
                data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                new_subset = _make_dace_subset_symbolic_context(
                    (
                        parent_symbolic_splits[sdfg.sdfg_id][0],
                        parent_symbolic_splits[sdfg.sdfg_id][1][name],
                    ),
                    access_info,
                    memlet.subset,
                    data_dims=data_dims,
                    tile_sizes=self.strides,
                )
            else:
                new_subset = edge.data.subset
            map_exit.add_in_connector("IN_" + name)
            map_exit.add_out_connector("OUT_" + name)
            nsdfg_state.add_edge(
                edge.src,
                edge.src_conn,
                map_exit,
                "IN_" + name,
                memlet=dace.Memlet(data=name, subset=new_subset),
            )
            nsdfg_state.add_edge(map_exit, "OUT_" + name, edge.dst, edge.dst_conn, memlet=edge.data)
            nsdfg_state.remove_edge(edge)

        # Safeguard: if you have no inputs/outputs still put an empty edge to
        # conserve topology
        if not nsdfg_state.edges_between(map_entry, nsdfg_node):
            nsdfg_state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
        if not nsdfg_state.edges_between(nsdfg_node, map_exit):
            nsdfg_state.add_edge(nsdfg_node, None, map_exit, None, dace.Memlet())


def get_all_child_states(cft: cf.ControlFlow):
    """Return a set of all states in the given control flow tree."""
    if isinstance(cft, cf.SingleState):
        return {cft.state}
    else:
        return set.union(*[get_all_child_states(e) for e in cft.children])


def _setup_match(sdfg, nodes, dims):
    subgraph = dace.sdfg.graph.SubgraphView(sdfg, nodes)
    partial_expansion = PartialExpansion()
    partial_expansion.setup_match(subgraph)
    partial_expansion.strides = {dcir.Axis(d): 8 for d in dims}
    return partial_expansion, subgraph


def _test_match(sdfg, states: Set[dace.SDFGState], dims: Sequence[str]):
    expansion, subgraph = _setup_match(sdfg, states, dims)
    return expansion.can_be_applied(sdfg, subgraph)


def _generate_matches(sd, cft: cf.ControlFlow, dims: Sequence[str]):
    subgraphs: List[Tuple[dace.SDFG, Set[dace.SDFGState]]] = list()
    candidate: Set[dace.SDFGState] = set()
    # reversed so that extent analysis tends to include stencil sinks
    for child in reversed(cft.children):
        if isinstance(child, cf.SingleState):
            if _test_match(sd, {child.state} | candidate, dims):
                candidate.add(child.state)
            else:
                if candidate:
                    subgraphs.append((sd, candidate))
                if _test_match(sd, {child.state}, dims):
                    candidate = {child.state}
                else:
                    candidate = set()
                    subsubgraphs = []
                    for nsdfg_node in child.state.nodes():
                        if isinstance(nsdfg_node, dace.nodes.NestedSDFG):
                            nsdfg_cft = cf.structured_control_flow_tree(
                                nsdfg_node.sdfg, lambda _: ""
                            )
                            subsubgraphs += _generate_matches(nsdfg_node.sdfg, nsdfg_cft, dims)
                    subgraphs.extend(subsubgraphs)
        else:
            if candidate:
                subgraphs.append((sd, candidate))
            candidate = set()
            subsubgraphs = []
            if isinstance(child, cf.IfScope):
                subsubgraphs += _generate_matches(sd, child.body, dims)
                if child.orelse is not None:
                    subsubgraphs += _generate_matches(sd, child.orelse, dims)
            elif isinstance(child, cf.ForScope):
                subsubgraphs += _generate_matches(sd, child.body, dims)
            if subsubgraphs:
                subgraphs.extend(subsubgraphs)
    if candidate:
        subgraphs.append((sd, candidate))
    return subgraphs


def partially_expand(sdfg: dace.SDFG, dims: Sequence[str]):
    """Utility function that greedily tries to do can_apply/apply
    on subgraphs, starting from the leafs upward
    """
    if len(dims) == 0:
        return

    cft = cf.structured_control_flow_tree(sdfg, lambda _: "")

    # Greedily identify subgraphs that can be tiled in dims
    matches = _generate_matches(sdfg, cft, dims=dims)

    for sd, nodes in matches:
        parent_sdfg = next(iter(nodes)).parent
        expansion, subgraph = _setup_match(sd, nodes, dims)
        expansion.apply(parent_sdfg)

    return sdfg
