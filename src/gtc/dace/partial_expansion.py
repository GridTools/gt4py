import copy
from typing import Dict, List, Set, Tuple, Union

import dace
import dace.codegen.control_flow as cf
import dace.sdfg.graph
import dace.symbolic
import pydantic
import sympy
from dace.properties import DictProperty, make_properties
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation import transformation
from dace.transformation.helpers import nest_sdfg_subgraph

import eve
from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gtc.dace.expansion_specification import Map, Skip
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import make_dace_subset, union_inout_memlets
from gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents


SymbolicOriginType = Tuple[dace.symbolic.SymbolicType, ...]
SymbolicDomainType = Dict[dcir.Axis, dace.symbolic.SymbolicType]


def _set_skips(node):
    expansion_specification = list(node.expansion_specification)
    expansion_specification[0] = Skip(item=expansion_specification[0])
    node.expansion_specification = expansion_specification


def _get_field_access_infos(
    state: dace.SDFGState,
    node: StencilComputation,
) -> Tuple[
    Dict[dcir.SymbolName, dcir.FieldAccessInfo], Dict[dcir.SymbolName, dcir.FieldAccessInfo]
]:
    """
    returns a dictionary mapping field names to access infos for the argument node with the set expansion_specification
    """

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


def _get_symbolic_origin_and_domain(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    node: StencilComputation,
    access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
) -> Tuple[Dict[int, Dict[dcir.SymbolName, SymbolicOriginType]], Dict[int, SymbolicDomainType]]:
    """
    match `node`'s in/out memlet subsets with the dcir.GridSubset in `access_infos` to determine which part of the
    memlet subset corresponds to halo and domain. returns a dict mapping the field names to a 2-tuple with
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

    origins = dict()
    access_infos = PartialExpansion._rename_dict_keys(
        PartialExpansion._get_name_map(state, node), access_infos
    )

    # Collect equations and symbols from arguments and shapes
    equations = []
    symbols = set()
    for field, info in access_infos.items():
        for axis in info.axes():
            if axis in info.variable_offset_axes:
                info = info.clamp_full_axis(axis)
        inner_exprs = [dace.symbolic.pystr_to_symbolic(s) for s in info.shape]
        outer_exprs = [
            dace.symbolic.pystr_to_symbolic(s) for s in outer_subsets[field].bounding_box_size()
        ]
        # strip data dims
        outer_exprs = outer_exprs[: len(inner_exprs)]

        for inner_expr, outer_expr in zip(inner_exprs, outer_exprs):
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
        raise ValueError("Symbolic split inconsistent.")
    result = results[0]
    result = {str(k)[len("__SOLVE_") :]: v for k, v in result.items()}
    domain = {
        axis: result[axis.domain_symbol()]
        for axis in dcir.Axis.dims_3d()
        if axis.domain_symbol() in result
    }
    domain.update(
        {
            ax: node.symbol_mapping[ax.domain_symbol()]
            for ax in dcir.Axis.dims_3d()
            if ax.domain_symbol() in node.symbol_mapping
        }
    )
    domains, origin_dicts = dict(), dict()
    for field, info in access_infos.items():
        outer_subset: dace.subsets.Range = outer_subsets[field]
        inner_origin = [
            info.grid_subset.intervals[axis].to_dace_symbolic()[0]
            for axis in dcir.Axis.dims_3d()
            if axis in info.grid_subset.intervals
        ]
        origin = list(
            dace.symbolic.pystr_to_symbolic(r[0] - o)
            for r, o in zip(outer_subset.ranges, inner_origin)
        )
        for i, orig in enumerate(origin):
            origin[i] = orig.subs(
                {ax.domain_dace_symbol(): value for ax, value in domain.items()}, simultaneous=True
            )
        origins[field] = tuple(origin)
        for ax_idx, axis in enumerate(info.axes()):
            interval_end = info.grid_subset.intervals[axis].end
            if interval_end.level == common.LevelMarker.END:
                domain.setdefault(
                    axis,
                    (outer_subset.ranges[ax_idx][1] + 1 - interval_end.offset)
                    - origins[field][ax_idx],
                )

    domains[state.parent.sdfg_id] = {**domains.get(state.parent.sdfg_id, domain), **domain}
    origin_dicts[state.parent.sdfg_id] = {**origin_dicts.get(state.parent.sdfg_id, origins), **origins}

    while state.parent.parent_sdfg is not None:
        tmp_origins = dict()
        tmp_domain = dict()
        nsdfg_node: dace.nodes.NestedSDFG = state.parent.parent_nsdfg_node

        for state in state.parent.parent_sdfg.states():
            if nsdfg_node in state.nodes():
                parent_state = state
                break
        else:
            raise AssertionError("NestedSDFG not found in parent SDFG.")
        for name, origin in origins.items():
            if name in nsdfg_node.in_connectors:
                outer_edge = next(
                    filter(lambda edge: edge.dst_conn == name, parent_state.in_edges(nsdfg_node))
                )
            elif name in nsdfg_node.out_connectors:
                outer_edge = next(
                    filter(lambda edge: edge.src_conn == name, parent_state.out_edges(nsdfg_node))
                )
            else:
                continue
            outer_name = outer_edge.data.data
            outer_origin = [r[0] for r in outer_edge.data.subset.ranges]
            inner_origin = [io.subs(nsdfg_node.symbol_mapping) for io in origins[name]]
            tmp_origins[outer_name] = [io + oo for io, oo in zip(inner_origin, outer_origin)]
        for ax, value in domain.items():
            tmp_domain[ax] = domain[ax].subs(nsdfg_node.symbol_mapping)
        domains[parent_state.parent.sdfg_id] = {**domains.get(parent_state.parent.sdfg_id, tmp_domain), **tmp_domain}
        origin_dicts[parent_state.parent.sdfg_id] = {**origin_dicts.get(parent_state.parent.sdfg_id, tmp_origins), **tmp_origins}

        state = parent_state
        origins = tmp_origins


    return origin_dicts, domains


def _union_symbolic_origin_and_domain(
    first: Tuple[Dict[int, Dict[dcir.SymbolName, SymbolicOriginType]], Dict[int, SymbolicDomainType]],
    second: Tuple[Dict[int, Dict[dcir.SymbolName, SymbolicOriginType]], Dict[int, SymbolicDomainType]],
):
    """
    union the key sets of the two dictionaries. (this utility is here so it can later be expanded to check consistency of the dictionaries.)
    """
    first_origins, first_domains = first
    second_origins, second_domains = second

    res_origins = {key: dict(value) for key, value in first_origins.items()}
    res_domains = {key: dict(value) for key, value in first_domains.items()}
    for sdfg_id, domain in second_domains.items():
        res_domains[sdfg_id] = {**res_domains.get(sdfg_id, domain), **domain}
    for sdfg_id, origins in second_origins.items():
        res_origins[sdfg_id] = {**res_origins.get(sdfg_id, origins), **origins}
    return res_origins, res_domains


def _union_access_infos(
    first: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
    second: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
) -> Dict[dcir.SymbolName, dcir.FieldAccessInfo]:
    res = dict(first)
    for key, value in second.items():
        res[key] = value.union(res.get(key, value))
    return res


def _get_access_infos(
    nodes: List[StencilComputation],
) -> Tuple[
    Dict[dcir.SymbolName, dcir.FieldAccessInfo], Dict[dcir.SymbolName, dcir.FieldAccessInfo]
]:
    in_access_infos = dict()
    out_access_infos = dict()
    for node, state in nodes:
        to_sdfg_name_map = PartialExpansion._get_name_map(state, node)
        in_node_access_infos, out_node_access_infos = _get_field_access_infos(state, node)
        in_node_access_infos = PartialExpansion._rename_dict_keys(
            to_sdfg_name_map, in_node_access_infos
        )
        out_node_access_infos = PartialExpansion._rename_dict_keys(
            to_sdfg_name_map, out_node_access_infos
        )
        in_access_infos = _union_access_infos(in_access_infos, in_node_access_infos)
        out_access_infos = _union_access_infos(out_access_infos, out_node_access_infos)
    return in_access_infos, out_access_infos


def _make_dace_subset_symbolic_context(
    context_info: Tuple[SymbolicOriginType, SymbolicDomainType],
    access_info: dcir.FieldAccessInfo,
    data_dims: Tuple[int, ...],
    tile_sizes,
) -> dace.subsets.Range:
    context_origin, context_domain = context_info
    clamped_access_info = access_info
    context_origin = list(context_origin)
    for ax_idx, axis in enumerate(access_info.axes()):
        if axis in access_info.variable_offset_axes:
            context_origin[ax_idx] = max(context_origin[ax_idx], 0)
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
    res_ranges = []

    for ax_idx, axis in enumerate(clamped_access_info.axes()):
        clamped_interval = clamped_access_info.grid_subset.intervals[axis]
        if isinstance(clamped_interval, dcir.DomainInterval):
            subset_start, subset_end = clamped_interval.start, clamped_interval.end
            subset_start_sym = context_origin[ax_idx] + subset_start.offset
            subset_end_sym = context_origin[ax_idx] + subset_end.offset
            if subset_start.level == common.LevelMarker.END:
                subset_start_sym += context_domain[axis]
            if subset_end.level == common.LevelMarker.END:
                subset_end_sym += context_domain[axis]
        else:
            subset_start, subset_end = clamped_interval.start_offset, clamped_interval.end_offset
            subset_start_sym = context_origin[ax_idx] + subset_start + axis.tile_dace_symbol()
            subset_end_sym = (
                context_origin[ax_idx] + subset_end + axis.tile_dace_symbol() + tile_sizes[axis]
            )
        res_ranges.append((subset_start_sym, subset_end_sym - 1, 1))
    res_ranges.extend((0, dim - 1, 1) for dim in data_dims)
    return dace.subsets.Range(res_ranges)


def _update_shapes(
    sdfg: dace.SDFG, access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo], domain_dict
):
    for name, access_info in access_infos.items():
        for axis in access_info.axes():
            if axis in access_info.variable_offset_axes:
                access_info = access_info.clamp_full_axis(axis)
        array = sdfg.arrays[name]
        if array.transient:
            shape = list(access_info.overapproximated_shape)
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
            shape = access_info.shape
            shape = [dace.symbolic.pystr_to_symbolic(s) for s in shape]
            for i, s in enumerate(shape):
                shape[i] = s.subs(
                    {ax.domain_dace_symbol(): value for ax, value in domain_dict.items()},
                    simultaneous=True,
                )
            ndim = len(shape)
            shape += list(array.shape[ndim:])
            array.shape = shape


@make_properties
class PartialExpansion(transformation.SubgraphTransformation):

    strides = DictProperty(
        key_type=dcir.Axis,
        value_type=int,
        default={dcir.Axis.I: 8, dcir.Axis.J: 8},
        desc="Map dimensions to partially expand to the corresponding map stride.",
    )

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

        # if any(edge.data.assignments for edge in subgraph.edges()):
        #     return False
        for nsdfg, _ in filter(
            lambda n: isinstance(n[0], dace.nodes.NestedSDFG),
            PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph),
        ):
            if not PartialExpansion._nsdfg_checks(nsdfg.sdfg):
                return False

        # E.g. Maps can't exist in subgraph
        if not all(
            isinstance(
                n,
                (dace.nodes.AccessNode, dace.SDFGState, StencilComputation, dace.nodes.NestedSDFG),
            )
            for n, _ in PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph)
        ):
            return False

        if any(
            isinstance(edge.src, dace.nodes.AccessNode)
            and isinstance(edge.dst, dace.nodes.AccessNode)
            for state in subgraph.nodes()
            for edge in state.edges()
        ):
            return False

        for node, _ in stencil_computations:
            if not isinstance(node.expansion_specification[0], Map) or not all(
                it.kind == "tiling" for it in node.expansion_specification[0].iterations
            ):
                return False

        if not PartialExpansion._test_symbolic_split(sdfg, stencil_computations):
            return False

        if not PartialExpansion._test_extents_compatible(stencil_computations):
            return False

        return True

    @staticmethod
    def _test_symbolic_split(
        sdfg: dace.SDFG,
        nodes: List[Tuple[StencilComputation, dace.SDFGState]],
    ) -> bool:

        symbolic_split_origins, symbolic_split_domains = dict(), dict()
        for node, state in nodes:
            access_infos = _union_access_infos(*_get_field_access_infos(state, node))
            node_symbolic_origins, node_symbolic_domains = _get_symbolic_origin_and_domain(
                sdfg, state, node, access_infos
            )
            node_symbolic_origin = node_symbolic_origins[sdfg.sdfg_id]
            node_symbolic_domain = node_symbolic_domains[sdfg.sdfg_id]

            if sdfg.sdfg_id in symbolic_split_domains and not all(
                (node_symbolic_domain[ax] == symbolic_split_domains[sdfg.sdfg_id][ax]) == True
                for ax in dcir.Axis.dims_horizontal()
                if ax in symbolic_split_domains[sdfg.sdfg_id] and ax in node_symbolic_domain
            ):
                return False

            # test origin consistency
            if sdfg.sdfg_id in symbolic_split_origins:
                common_names = {
                    name for name in node_symbolic_origin.keys() if name in symbolic_split_origins[sdfg.sdfg_id]
                }
            else:
                common_names = set()

            for name in common_names:
                assert len(node_symbolic_origin[name]) == len(symbolic_split_origins[sdfg.sdfg_id][name])
                if not all(
                    (node_orig == res_orig) == True
                    for node_orig, res_orig in zip(
                        node_symbolic_origin[name], symbolic_split_origins[sdfg.sdfg_id][name]
                    )
                ):
                    return False

            symbolic_split_origins, symbolic_split_domains = _union_symbolic_origin_and_domain(
                (symbolic_split_origins, symbolic_split_domains),
                (node_symbolic_origins, node_symbolic_domains),
            )
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

    @staticmethod
    def _test_extents_compatible(
        nodes: List[Tuple[StencilComputation, dace.SDFGState]],
    ) -> bool:
        vertical_loops = []
        declarations = {}
        for node, state in nodes:
            name_map = PartialExpansion._get_name_map(state, node)
            vl = PartialExpansion._Renamer(name_map).visit(node.oir_node)
            vertical_loops.append(vl)

            for name, decl in filter(
                lambda n: isinstance(n[1], oir.FieldDecl), node.declarations.items()
            ):
                declarations[name_map[name]] = oir.FieldDecl(
                    name=name_map[name],
                    dtype=decl.dtype,
                    dimensions=decl.dimensions,
                    data_dims=decl.data_dims,
                )
        for node, _ in nodes:
            if (
                len(node.oir_node.iter_tree().if_isinstance(oir.HorizontalRestriction).to_list())
                > 0
            ):
                return False
        params = {
            name: decl
            for n, _ in nodes
            for name, decl in n.declarations.items()
            if isinstance(decl, oir.ScalarDecl)
        }

        oir_stencil = oir.Stencil(
            name="__adhoc",
            vertical_loops=vertical_loops,
            params=list(params.values()),
            declarations=list(declarations.values()),
        )
        oir_extents = compute_horizontal_block_extents(oir_stencil)

        for i, (node, _) in enumerate(nodes):
            for j, section in enumerate(node.oir_node.sections):
                for k, he in enumerate(section.horizontal_executions):
                    he_id = id(vertical_loops[i].sections[j].horizontal_executions[k])
                    if not oir_extents[he_id] == node.get_extents(he):
                        return False
        return True

    @staticmethod
    def _get_symbolic_splits(
        sdfg: dace.SDFG,
        nodes: List[Tuple[StencilComputation, dace.SDFGState]],
    ) -> Tuple[Dict[int, Dict[dcir.SymbolName, SymbolicOriginType]], Dict[int, SymbolicDomainType]]:

        symbolic_split = dict(), dict()
        for node, state in nodes:
            access_infos = _union_access_infos(*_get_field_access_infos(state, node))
            node_symbolic_split = _get_symbolic_origin_and_domain(sdfg, state, node, access_infos)
            symbolic_split = _union_symbolic_origin_and_domain(symbolic_split, node_symbolic_split)
        return symbolic_split

    @staticmethod
    def _get_name_map(state: dace.SDFGState, node: StencilComputation):
        res = dict()
        for edge in filter(lambda e: not e.data.is_empty(), state.in_edges(node)):
            inner_name = edge.dst_conn[len("__in_") :]
            outer_name = edge.data.data
            res[inner_name] = outer_name
        for edge in filter(lambda e: not e.data.is_empty(), state.out_edges(node)):
            inner_name = edge.src_conn[len("__out_") :]
            outer_name = edge.data.data
            res[inner_name] = outer_name
        return res

    @staticmethod
    def _rename_dict_keys(name_map, input_dict):
        return {name_map.get(key, key): value for key, value in input_dict.items()}

    @staticmethod
    def _update_nested_and_gather_access_info(nsdfg_node, domain_dicts, axes):
        all_access_infos = dict()
        nsdfg_access_infos = dict()
        for state in nsdfg_node.sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    node_access_infos = PartialExpansion._update_nested_and_gather_access_info(
                        node, domain_dicts, axes
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
        #
        # # in_nsdfg_access_infos, out_nsdfg_access_infos = _get_access_infos(stencil_computations)
        # # nsdfg_access_infos = _union_access_infos(in_nsdfg_access_infos, out_nsdfg_access_infos)

        # update shapes and subsets of new nested SDFG

        _update_shapes(nsdfg_node.sdfg, nsdfg_access_infos, domain_dicts[nsdfg_node.sdfg.sdfg_id])

        for state in nsdfg_node.sdfg.states():
            for node in filter(
                lambda n: isinstance(n, (dace.nodes.NestedSDFG, StencilComputation)), state.nodes()
            ):
                # to_sdfg_name_map = PartialExpansion._get_name_map(state, node)
                # in_access_infos, out_access_infos = _get_field_access_infos(state, node)
                # node.access_infos = _union_access_infos(in_access_infos, out_access_infos)
                # in_access_infos = PartialExpansion._rename_dict_keys(to_sdfg_name_map, in_access_infos)
                # out_access_infos = PartialExpansion._rename_dict_keys(to_sdfg_name_map, out_access_infos)
                node_access_infos, in_access_infos, out_access_infos = all_access_infos[node]
                if isinstance(node, StencilComputation):
                    node.access_infos = PartialExpansion._rename_dict_keys(
                        {v: k for k, v in PartialExpansion._get_name_map(state, node).items()},
                        node_access_infos,
                    )
                for memlet in state.in_edges(node):
                    name = memlet.data.data
                    access_info = in_access_infos[name]
                    naxes = len(access_info.grid_subset.intervals)
                    data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                    subset = make_dace_subset(
                        nsdfg_access_infos[name], access_info, data_dims=data_dims
                    )
                    subset.replace({ax.domain_dace_symbol(): v for ax, v in domain_dicts[state.parent.sdfg_id].items()})
                    memlet.data.subset = subset
                for memlet in state.out_edges(node):
                    name = memlet.data.data
                    access_info = out_access_infos[name]
                    naxes = len(access_info.grid_subset.intervals)
                    data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                    subset = memlet.data.subset = make_dace_subset(
                        nsdfg_access_infos[name], access_info, data_dims=data_dims
                    )
                    subset.replace({ax.domain_dace_symbol(): v for ax, v in domain_dicts[state.parent.sdfg_id].items()})
                    memlet.data.subset = subset

        nsdfg_node.symbol_mapping.update({ax.tile_symbol(): ax.tile_dace_symbol() for ax in axes})
        for ax in axes:
            if ax.tile_symbol() not in nsdfg_node.sdfg.symbols:
                nsdfg_node.sdfg.add_symbol(ax.tile_symbol(), stype=dace.int32)

        return nsdfg_access_infos

    def apply(self, sdfg: dace.SDFG):
        subgraph = self.subgraph_view(sdfg)

        stencil_computations: List[StencilComputation, dace.SDFGState] = list(
            filter(
                lambda n: isinstance(n[0], StencilComputation),
                PartialExpansion.subgraph_all_nodes_recursive_topological(subgraph),
            )
        )

        first_expansion_specification = stencil_computations[0][0].expansion_specification

        original_item = first_expansion_specification[0]

        parent_symbolic_splits = PartialExpansion._get_symbolic_splits(sdfg, stencil_computations)
        domain_dict = parent_symbolic_splits[1][sdfg.sdfg_id]
        tiling_schedule = original_item.schedule
        tiling_ranges = {
            it.axis.tile_symbol(): str(
                dace.subsets.Range([(0, domain_dict[it.axis] - 1, it.stride)])
            )
            for it in original_item.iterations
        }
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
        for node, _ in stencil_computations:
            _set_skips(node)

        axes = {it.axis for it in original_item.iterations}
        nsdfg_access_infos = PartialExpansion._update_nested_and_gather_access_info(
            nsdfg_node, parent_symbolic_splits[1], axes
        )

        propagate_memlets_sdfg(nsdfg_node.sdfg)

        map_entry, map_exit = nsdfg_state.add_map(
            stencil_computations[0][0].label + "_tiling_map",
            ndrange=tiling_ranges,
            schedule=tiling_schedule,
            debuginfo=nsdfg_node.debuginfo,
        )
        # nsdfg_node.symbol_mapping.update(
        #     {it.axis.tile_symbol(): it.axis.tile_dace_symbol() for it in original_item.iterations}
        # )
        # for it in original_item.iterations:
        #     if it.axis.tile_symbol() not in nsdfg_node.sdfg.symbols:
        #         nsdfg_node.sdfg.add_symbol(it.axis.tile_symbol(), stype=dace.int32)
        for edge in nsdfg_state.in_edges(nsdfg_node):
            memlet = edge.data
            name = memlet.data
            if name in nsdfg_access_infos:
                access_info = nsdfg_access_infos[name]
                naxes = len(access_info.grid_subset.intervals)
                data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                new_subset = _make_dace_subset_symbolic_context(
                    (parent_symbolic_splits[0][sdfg.sdfg_id][name], parent_symbolic_splits[1][sdfg.sdfg_id]),
                    access_info,
                    data_dims=data_dims,
                    tile_sizes={it.axis: it.stride for it in original_item.iterations},
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
                    (parent_symbolic_splits[0][sdfg.sdfg_id][name], parent_symbolic_splits[1][sdfg.sdfg_id]),
                    access_info,
                    data_dims=data_dims,
                    tile_sizes={it.axis: it.stride for it in original_item.iterations},
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
        if not nsdfg_state.edges_between(map_entry, nsdfg_node):
            nsdfg_state.add_edge(map_entry, None, nsdfg_node, None, dace.Memlet())
        if not nsdfg_state.edges_between(nsdfg_node, map_exit):
            nsdfg_state.add_edge(nsdfg_node, None, map_exit, None, dace.Memlet())


def get_all_child_states(cft: cf.ControlFlow):
    """Returns a set of all states in the given control flow tree."""
    if isinstance(cft, cf.SingleState):
        return {cft.state}
    else:
        return set.union(*[get_all_child_states(e) for e in cft.children])


def _setup_match(sdfg, nodes):
    subgraph = dace.sdfg.graph.SubgraphView(sdfg, nodes)
    partial_expansion = PartialExpansion()
    partial_expansion.setup_match(subgraph)
    partial_expansion.strides = {dcir.Axis.I: 8, dcir.Axis.J: 8}
    return partial_expansion, subgraph


def _test_match(sdfg, states: Set[dace.SDFGState]):
    expansion, subgraph = _setup_match(sdfg, states)
    return expansion.can_be_applied(sdfg, subgraph)


def partially_expand(sdfg: dace.SDFG):
    matches: List[Tuple[dace.SDFG, Set[dace.SDFGState]]] = []
    # for sd in sdfg.all_sdfgs_recursive():
    cft = cf.structured_control_flow_tree(sdfg, lambda _: "")

    def _recurse(sd, cft: cf.ControlFlow):
        subgraphs: List[Tuple[dace.SDFG, Set[dace.SDFGState]]] = list()
        candidate = set()
        # reversed so that extent analysis tends to include stencil sinks
        for child in reversed(cft.children):
            if isinstance(child, cf.SingleState):
                if _test_match(sd, {child.state} | candidate):
                    candidate.add(child.state)
                else:
                    if candidate:
                        subgraphs.append((sd, candidate))
                    if _test_match(sd, {child.state}):
                        candidate = {child.state}
                    else:
                        candidate = set()
                        subsubgraphs = []
                        for nsdfg_node in child.state.nodes():
                            if isinstance(nsdfg_node, dace.nodes.NestedSDFG):
                                nsdfg_cft = cf.structured_control_flow_tree(
                                    nsdfg_node.sdfg, lambda _: ""
                                )
                                subsubgraphs += _recurse(nsdfg_node.sdfg, nsdfg_cft)
                        subgraphs.extend(subsubgraphs)
            else:
                if candidate:
                    subgraphs.append((sd, candidate))
                candidate = set()
                subsubgraphs = []
                if isinstance(child, cf.IfScope):
                    subsubgraphs += _recurse(sd, child.body)
                    if child.orelse is not None:
                        subsubgraphs += _recurse(sd, child.orelse)
                elif isinstance(child, cf.ForScope):
                    subsubgraphs += _recurse(sd, child.body)
                if subsubgraphs:
                    subgraphs.extend(subsubgraphs)
        if candidate:
            subgraphs.append((sd, candidate))
        return subgraphs

    matches = _recurse(sdfg, cft)

    for sd, nodes in matches:
        parent_sdfg = next(iter(nodes)).parent
        expansion, subgraph = _setup_match(sd, nodes)
        expansion.apply(parent_sdfg)

    # assert subgraphs
    # allstates = [s for subgraph in subgraphs for s in subgraph]
    # print(subgraphs)
    # assert set(allstates) == set(sdfg.nodes())  # every state is here
    # assert len(allstates) == sdfg.number_of_nodes()  # all states are here once

    return sdfg
