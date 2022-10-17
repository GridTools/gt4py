import dace
from dace.transformation.dataflow import TrivialMapElimination
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.interstate import InlineTransients


def eliminate_trivial_maps(sdfg: dace.SDFG):
    """Remove maps and map ranges where the iteration is over a single index."""
    applied = True
    while applied:
        applied = False
        for map_entry, state in sdfg.all_nodes_recursive():
            if isinstance(map_entry, dace.nodes.MapEntry):
                if map_entry.map.schedule in {
                    dace.ScheduleType.Sequential,
                    dace.ScheduleType.CPU_Multicore,
                }:
                    # exclude maps with empty edges as workaround for a bug in TrivialMapElimination
                    if any(
                        edge.data.data is None
                        for edge in state.in_edges(map_entry)
                        + state.out_edges(state.exit_node(map_entry))
                    ):
                        continue
                    try:
                        TrivialMapElimination.apply_to(
                            state.parent, map_entry=map_entry, verify=True
                        )
                        applied = True
                        break
                    except ValueError:
                        continue


class InlineThreadLocalTransients(dace.transformation.SingleStateTransformation):
    """
    Inline and tile thread-local transients.

    Inlines transients like `dace.transformations.interstate.InlineTransients`, however only applies to OpenMP map
    scopes but also makes the resulting local arrays persistent and thread-local. This reproduces `cpu_kfirst`-style
    transient tiling.
    """

    map_entry = dace.transformation.transformation.PatternNode(dace.nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        map_entry = self.map_entry

        if not map_entry.schedule == dace.ScheduleType.CPU_Multicore:
            return False

        scope_subgraph = graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        if len(scope_subgraph) > 1 or not isinstance(
            scope_subgraph.nodes()[0], dace.nodes.NestedSDFG
        ):
            return False

        toremove = InlineTransients._candidates(sdfg, graph, scope_subgraph.nodes()[0])
        return len(toremove) > 0

    def apply(self, graph, sdfg):
        map_entry = self.map_entry

        scope_subgraph = graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)
        nsdfg_node = scope_subgraph.nodes()[0]
        toremove = InlineTransients._candidates(sdfg, graph, nsdfg_node)
        InlineTransients.apply_to(sdfg, nsdfg=nsdfg_node)
        for name in toremove:
            array: dace.data.Array = nsdfg_node.sdfg.arrays[name]
            shape = [dace.symbolic.overapproximate(s) for s in array.shape]
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


def nest_sequential_map_scopes(sdfg: dace.SDFG):
    """Nest map scopes of sequential maps.

    Nest scope subgraphs of sequential maps in NestedSDFG's to force eagerly offsetting pointers on each iteration, to
    avoid more complex pointer arithmetic on each Tasklet's invocation.
    This is performed in an inner-map-first order to avoid revisiting the graph after changes.
    """
    visited = set()

    def _process_map(sdfg: dace.SDFG, state: dace.SDFGState, map_entry: dace.nodes.MapEntry):
        for node in state.scope_subgraph(map_entry, include_entry=False, include_exit=False):
            if node in visited:
                continue
            if isinstance(node, dace.nodes.NestedSDFG):
                nest_sequential_map_scopes(node.sdfg)
            elif isinstance(node, dace.nodes.MapEntry):
                _process_map(sdfg, state, node)
            visited.add(node)
        if map_entry.schedule == dace.ScheduleType.Sequential:
            map_entrys = [map_entry]
            for me in reversed(map_entrys):
                subgraph = state.scope_subgraph(me, include_entry=False, include_exit=False)
                nest_state_subgraph(sdfg, state, subgraph)

    for state in sdfg.nodes():
        for map_entry in filter(
            lambda n: isinstance(n, dace.nodes.MapEntry) and n not in visited, state.nodes()
        ):
            _process_map(sdfg, state, map_entry)
            visited.add(map_entry)
