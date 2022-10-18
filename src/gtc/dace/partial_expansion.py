from typing import Dict, List, Tuple

import dace.sdfg.graph
import dace.symbolic
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.helpers import nest_sdfg_subgraph

from gtc import common
from gtc import daceir as dcir
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import make_dace_subset


SymbolicOriginType = Tuple[dace.symbolic.SymbolicType, ...]
SymbolicDomainType = Tuple[dace.symbolic.SymbolicType, ...]


def _set_skips(node, expansion_items):
    raise NotImplementedError


def _get_field_access_infos(
    node: StencilComputation,
) -> Tuple[
    Dict[dcir.SymbolName, dcir.FieldAccessInfo], Dict[dcir.SymbolName, dcir.FieldAccessInfo]
]:
    """
    returns a dictionary mapping field names to access infos for the argument node with the set expansion_specification
    """
    raise NotImplementedError


def _get_symbolic_origin_and_domain(
    node: StencilComputation, access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo]
) -> Dict[dcir.SymbolName, Tuple[SymbolicOriginType, SymbolicDomainType]]:
    """
    match `node`'s in/out memlet subsets with the dcir.GridSubset in `access_infos` to determine which part of the
    memlet subset corresponds to halo and domain. returns a dict mapping the field names to a 2-tuple with
    * the origin of the field in the node as absolute index in the parent sdfg's array
    * the shape of the domain as tuple of symbolic values
    """
    raise NotImplementedError


def _union_symbolic_origin_and_domain(
    first: Dict[dcir.SymbolName, Tuple[SymbolicOriginType, SymbolicDomainType]],
    second: Dict[dcir.SymbolName, Tuple[SymbolicOriginType, SymbolicDomainType]],
):
    """
    union the key sets of the two dictionaries. (this utility is here so it can later be expanded to check consistency of the dictionaries.)
    """
    res = dict(first)
    res.update(**second)
    return res


def _union_access_infos(
    first: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
    second: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
) -> Dict[dcir.SymbolName, dcir.FieldAccessInfo]:
    res = dict(first)
    for key, value in second.items():
        res[key] = value.union(res.get(key, value))
    return res


def _get_symbolic_split(
    nodes: List[StencilComputation],
) -> Dict[dcir.SymbolName, Tuple[SymbolicOriginType, SymbolicDomainType]]:
    symbolic_split = dict()
    for node, _ in nodes:
        in_node_access_infos, out_node_access_infos = _get_field_access_infos(node)
        node_symbolic_split = _union_symbolic_origin_and_domain(
            _get_symbolic_origin_and_domain(node, in_node_access_infos),
            _get_symbolic_origin_and_domain(node, out_node_access_infos),
        )
        symbolic_split = _union_symbolic_origin_and_domain(symbolic_split, node_symbolic_split)
    return symbolic_split


def _get_access_infos(
    nodes: List[StencilComputation],
) -> Tuple[
    Dict[dcir.SymbolName, dcir.FieldAccessInfo], Dict[dcir.SymbolName, dcir.FieldAccessInfo]
]:
    in_access_infos = dict()
    out_access_infos = dict()
    for node, _ in nodes:
        in_node_access_infos, out_node_access_infos = _get_field_access_infos(node)
        in_access_infos, out_access_infos = _union_access_infos(
            in_access_infos, in_node_access_infos
        ), _union_access_infos(out_access_infos, out_node_access_infos)
    return in_access_infos, out_access_infos


def _make_dace_subset_symbolic_context(
    context_info: Tuple[SymbolicOriginType, SymbolicDomainType],
    access_info: dcir.FieldAccessInfo,
    data_dims: Tuple[int, ...],
) -> dace.subsets.Range:
    context_origin, context_domain = context_info
    clamped_access_info = access_info
    for axis in access_info.axes():
        if axis in access_info.variable_offset_axes:
            clamped_access_info = clamped_access_info.clamp_full_axis(axis)
    res_ranges = []

    for ax_idx, axis in enumerate(clamped_access_info.axes()):
        clamped_interval = clamped_access_info.grid_subset.intervals[axis]
        subset_start, subset_end = clamped_interval.start, clamped_interval.end
        subset_start_sym = context_origin[ax_idx] + subset_start.offset
        subset_end_sym = context_origin[ax_idx] + subset_end.offset
        if subset_start.level == common.LevelMarker.END:
            subset_start_sym += context_domain[ax_idx]
        if subset_end.level == common.LevelMarker.END:
            subset_end_sym += context_domain[ax_idx]
        res_ranges.append((subset_start_sym, subset_end_sym - 1, 1))
    res_ranges.extend((0, dim - 1, 1) for dim in data_dims)
    return dace.subsets.Range(res_ranges)


def _update_shapes(sdfg, access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo]):
    for name, access_info in access_infos.items():
        array = sdfg.arrays[name]
        array.shape = access_info.shape


def partially_expand(sdfg, expansion_items):

    stencil_computations: List[StencilComputation] = list(
        filter(lambda n: isinstance(n[0], StencilComputation), sdfg.all_nodes_recursive())
    )

    original_item = stencil_computations[0][0].expansion_specification[0]
    tiling_schedule = original_item.schedule
    # for iteration in original_item.iterations:
    #     axis: dcir.Axis = iteration.axis
    #     stride = iteration.stride
    #     start, end = ...
    #      {axis.tile_dace_symbol(): str(dace.subsets.Range([(start, end - 1, stride)]))}
    #     limits = ...

    parent_symbolic_split = _get_symbolic_split(stencil_computations)

    nsdfg_state: dace.SDFGState = nest_sdfg_subgraph(
        dace.sdfg.graph.SubgraphView(sdfg.subgraphZ, list(sdfg.states()))
    )
    nsdfg_node: dace.nodes.NestedSDFG = next(
        iter(node for node in nsdfg_state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    )

    for node in stencil_computations:
        _set_skips(node, expansion_items)

    in_nsdfg_access_infos, out_nsdfg_access_infos = _get_access_infos(stencil_computations)
    nsdfg_access_infos = _union_access_infos(in_nsdfg_access_infos, out_nsdfg_access_infos)

    # update shapes and subsets of new nested SDFG
    _update_shapes(nsdfg_node.sdfg, nsdfg_access_infos)
    for state in nsdfg_node.sdfg.states():
        for node in filter(lambda n: isinstance(n, StencilComputation), state.nodes()):
            in_access_infos, out_access_infos = _get_access_infos(node)
            for memlet in state.in_edges(node):
                name = memlet.data.data
                access_info = in_access_infos[name]
                naxes = len(access_info.grid_subset.intervals)
                data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                memlet.data.subset = make_dace_subset(
                    nsdfg_access_infos[name], access_info, data_dims=data_dims
                )
            for memlet in state.out_edges(node):
                name = memlet.data.data
                access_info = out_access_infos[name]
                naxes = len(access_info.grid_subset.intervals)
                data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
                memlet.data.subset = make_dace_subset(
                    nsdfg_access_infos[name], access_info, data_dims=data_dims
                )
    propagate_memlets_sdfg(nsdfg_node.sdfg)

    map_entry, map_exit = nsdfg_state.add_map(
        stencil_computations[0].label + "_tiling_map",
        ndrange=tiling_ranges,
        schedule=tiling_schedule,
        debuginfo=nsdfg_node.debuginfo,
    )
    for edge in nsdfg_state.in_edges(nsdfg_node):
        memlet = edge.data
        name = memlet.data
        access_info = nsdfg_access_infos[name]
        naxes = len(access_info.grid_subset.intervals)
        data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
        map_entry.add_in_connector("IN_" + name)
        map_entry.add_out_connector("OUT_" + name)
        new_subset = _make_dace_subset_symbolic_context(
            parent_symbolic_split[name], access_info, data_dims=data_dims
        )
        nsdfg_state.add_edge(edge.src, edge.src_conn, map_entry, "IN_" + name, memlet=edge.memlet)
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
        access_info = nsdfg_access_infos[name]
        naxes = len(access_info.grid_subset.intervals)
        data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
        map_exit.add_in_connector("IN_" + name)
        map_exit.add_out_connector("OUT_" + name)
        new_subset = _make_dace_subset_symbolic_context(
            parent_symbolic_split[name], access_info, data_dims=data_dims
        )
        nsdfg_state.add_edge(
            edge.src,
            edge.src_conn,
            map_exit,
            "IN_" + name,
            memlet=dace.Memlet(data=name, subset=new_subset),
        )
        nsdfg_state.add_edge(
            map_entry, "OUT_" + name, nsdfg_node, edge.dst_conn, memlet=edge.memlet
        )
        nsdfg_state.remove_edge(edge)
