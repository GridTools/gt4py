import copy
from typing import Dict, List, Tuple

import dace.sdfg.graph
import dace.symbolic
import sympy
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.helpers import nest_sdfg_subgraph

from gtc import common
from gtc import daceir as dcir
from gtc.dace.expansion.daceir_builder import DaCeIRBuilder
from gtc.dace.expansion_specification import Map, Skip
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import compute_dcir_access_infos, make_dace_subset, union_inout_memlets


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
    state: dace.SDFGState,
    node: StencilComputation,
    access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
) -> Tuple[Dict[dcir.SymbolName, SymbolicOriginType], SymbolicDomainType]:
    """
    match `node`'s in/out memlet subsets with the dcir.GridSubset in `access_infos` to determine which part of the
    memlet subset corresponds to halo and domain. returns a dict mapping the field names to a 2-tuple with
    * the origin of the field in the node as absolute index in the parent sdfg's array
    * the shape of the domain as tuple of symbolic values
    """
    # union input and output subsets
    outer_subsets = {}
    for edge in state.in_edges(node):
        outer_subsets[edge.dst_conn[len("__in_") :]] = edge.data.subset
    for edge in state.out_edges(node):
        outer_subsets[edge.src_conn[len("__out_") :]] = dace.subsets.union(
            edge.data.subset, outer_subsets.get(edge.src_conn[len("__out_") :], edge.data.subset)
        )
    # ensure single-use of input and output subset instances
    for edge in state.in_edges(node):
        edge.data.subset = copy.deepcopy(outer_subsets[edge.dst_conn[len("__in_") :]])
    for edge in state.out_edges(node):
        edge.data.subset = copy.deepcopy(outer_subsets[edge.src_conn[len("__out_") :]])

    origins = dict()
    domain = dict()
    for field, info in access_infos.items():
        outer_subset: dace.subsets.Range = outer_subsets[field]
        inner_origin = [
            info.grid_subset.intervals[axis].to_dace_symbolic()[0]
            for axis in dcir.Axis.dims_3d()
            if axis in info.grid_subset.intervals
        ]
        origins[field] = tuple(
            dace.symbolic.pystr_to_symbolic(r[0] - o)
            for r, o in zip(outer_subset.ranges, inner_origin)
        )
        for ax_idx, axis in enumerate(info.axes()):
            interval_end = info.grid_subset.intervals[axis].end
            if interval_end.level == common.LevelMarker.END:
                domain[axis] = (outer_subset.ranges[ax_idx][1] + 1 - interval_end.offset) - origins[
                    field
                ][ax_idx]
    for axis in dcir.Axis.dims_3d():
        domain.setdefault(axis, 0)

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
    result = results[0]
    result = {str(k)[len("__SOLVE_") :]: v for k, v in result.items()}
    solved_domain = {
        axis: result[axis.domain_symbol()]
        for axis in dcir.Axis.dims_3d()
        if axis.domain_symbol() in result
    }
    domain.update(**solved_domain)
    return origins, domain


def _union_symbolic_origin_and_domain(
    first: Tuple[Dict[dcir.SymbolName, SymbolicOriginType], SymbolicDomainType],
    second: Tuple[Dict[dcir.SymbolName, SymbolicOriginType], SymbolicDomainType],
):
    """
    union the key sets of the two dictionaries. (this utility is here so it can later be expanded to check consistency of the dictionaries.)
    """
    res_origins = dict(first[0])
    res_origins.update(**second[0])
    res_domain = dict(first[1])
    res_domain.update(**second[1])
    return res_origins, res_domain


def _union_access_infos(
    first: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
    second: Dict[dcir.SymbolName, dcir.FieldAccessInfo],
) -> Dict[dcir.SymbolName, dcir.FieldAccessInfo]:
    res = dict(first)
    for key, value in second.items():
        res[key] = value.union(res.get(key, value))
    return res


def _get_symbolic_split(
    nodes: List[Tuple[StencilComputation, dace.SDFGState]],
) -> Tuple[Dict[dcir.SymbolName, SymbolicOriginType], Dict[dcir.Axis, SymbolicDomainType]]:

    symbolic_split = dict(), dict()
    for node, state in nodes:
        access_infos = _union_access_infos(*_get_field_access_infos(state, node))
        node_symbolic_split = _get_symbolic_origin_and_domain(state, node, access_infos)
        symbolic_split = _union_symbolic_origin_and_domain(symbolic_split, node_symbolic_split)
    return symbolic_split


def _get_access_infos(
    nodes: List[StencilComputation],
) -> Tuple[
    Dict[dcir.SymbolName, dcir.FieldAccessInfo], Dict[dcir.SymbolName, dcir.FieldAccessInfo]
]:
    in_access_infos = dict()
    out_access_infos = dict()
    for node, state in nodes:
        in_node_access_infos, out_node_access_infos = _get_field_access_infos(state, node)
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


def _update_shapes(sdfg: dace.SDFG, access_infos: Dict[dcir.SymbolName, dcir.FieldAccessInfo]):
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
            ndim = len(shape)
            shape += list(array.shape[ndim:])
            array.shape = shape


def partially_expand(sdfg):
    stencil_computations: List[StencilComputation, dace.SDFGState] = list(
        filter(lambda n: isinstance(n[0], StencilComputation), sdfg.all_nodes_recursive())
    )
    if not stencil_computations:
        return

    original_item = stencil_computations[0][0].expansion_specification[0]
    for node, _ in stencil_computations:
        if not isinstance(node.expansion_specification[0], Map) or not all(
            it.kind == "tiling" for it in node.expansion_specification[0].iterations
        ):
            return

    tiling_schedule = original_item.schedule
    tiling_ranges = {
        it.axis.tile_symbol(): str(
            dace.subsets.Range([(0, it.axis.domain_dace_symbol() - 1, it.stride)])
        )
        for it in original_item.iterations
    }

    parent_symbolic_split = _get_symbolic_split(stencil_computations)

    states = list(sdfg.states())
    sdfg: dace.SDFG
    if len(states) == 1:
        new_state = sdfg.add_state_before(states[0])
        sdfg.validate()
        states = [new_state, *states]
    nsdfg_state: dace.SDFGState = nest_sdfg_subgraph(
        sdfg, dace.sdfg.graph.SubgraphView(sdfg, states)
    )
    nsdfg_node: dace.nodes.NestedSDFG = next(
        iter(node for node in nsdfg_state.nodes() if isinstance(node, dace.nodes.NestedSDFG))
    )

    for node, _ in stencil_computations:
        _set_skips(node)

    in_nsdfg_access_infos, out_nsdfg_access_infos = _get_access_infos(stencil_computations)
    nsdfg_access_infos = _union_access_infos(in_nsdfg_access_infos, out_nsdfg_access_infos)

    # update shapes and subsets of new nested SDFG
    _update_shapes(nsdfg_node.sdfg, nsdfg_access_infos)
    for state in nsdfg_node.sdfg.states():
        for node in filter(lambda n: isinstance(n, StencilComputation), state.nodes()):
            in_access_infos, out_access_infos = _get_field_access_infos(state, node)
            node.access_infos = _union_access_infos(in_access_infos, out_access_infos)
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
        stencil_computations[0][0].label + "_tiling_map",
        ndrange=tiling_ranges,
        schedule=tiling_schedule,
        debuginfo=nsdfg_node.debuginfo,
    )
    nsdfg_node.symbol_mapping.update(
        {it.axis.tile_symbol(): it.axis.tile_dace_symbol() for it in original_item.iterations}
    )
    for it in original_item.iterations:
        if it.axis.tile_symbol() not in nsdfg_node.sdfg.symbols:
            nsdfg_node.sdfg.add_symbol(it.axis.tile_symbol(), stype=dace.int32)
    for edge in nsdfg_state.in_edges(nsdfg_node):
        memlet = edge.data
        name = memlet.data
        access_info = nsdfg_access_infos[name]
        naxes = len(access_info.grid_subset.intervals)
        data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
        map_entry.add_in_connector("IN_" + name)
        map_entry.add_out_connector("OUT_" + name)
        new_subset = _make_dace_subset_symbolic_context(
            (parent_symbolic_split[0][name], parent_symbolic_split[1]),
            access_info,
            data_dims=data_dims,
            tile_sizes={it.axis: it.stride for it in original_item.iterations},
        )
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
        access_info = nsdfg_access_infos[name]
        naxes = len(access_info.grid_subset.intervals)
        data_dims = nsdfg_node.sdfg.arrays[name].shape[naxes:]
        map_exit.add_in_connector("IN_" + name)
        map_exit.add_out_connector("OUT_" + name)
        new_subset = _make_dace_subset_symbolic_context(
            (parent_symbolic_split[0][name], parent_symbolic_split[1]),
            access_info,
            data_dims=data_dims,
            tile_sizes={it.axis: it.stride for it in original_item.iterations},
        )
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
