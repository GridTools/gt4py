import dace
from dace import registry
from dace.sdfg import SDFG


def global_ij_tiling(sdfg, tile_size=(8, 8)):
    input_arrays = dict()
    output_arrays = dict()
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if (
                    node.access is dace.AccessType.ReadOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = input_arrays.get(node.data, 0)
                    input_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.out_edges(node)]
                    )

                if (
                    node.access is dace.AccessType.WriteOnly
                    or node.access is dace.AccessType.ReadWrite
                ) and not sdfg.arrays[node.data].transient:
                    num_accesses = output_arrays.get(node.data, 0)
                    output_arrays[node.data] = num_accesses + sum(
                        [e.data.num_accesses for e in state.in_edges(node)]
                    )

    # nest state
    import copy

    tmp_sdfg = copy.deepcopy(sdfg)
    for s in sdfg.nodes():
        sdfg.remove_node(s)
    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(
        tmp_sdfg, sdfg, list(input_arrays.keys()), list(output_arrays.keys())
    )
    nsdfg_node.symbol_mapping.update(
        # I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i*{tile_size[0]})"),
        # J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j*{tile_size[1]})"),
        I=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[0]}, I-tile_i)"),
        J=dace.symbolic.pystr_to_symbolic(f"Min({tile_size[1]}, J-tile_j)"),
    )
    # map
    map_entry, map_exit = state.add_map(
        "global_tiling",
        ndrange=dict(
            # tile_i=f"0:int_ceil(I, {tile_size[0]})", tile_j=f"0:int_ceil(J, {tile_size[1]})"
            tile_i=f"0:I:{tile_size[0]}",
            tile_j=f"0:J:{tile_size[1]}",
        ),
    )
    map_entry.map.collapse = 2

    # conn_id = 0
    for array_name, num_accesses in input_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_entry.add_in_connector("IN_" + array_name)
            map_entry.add_out_connector("OUT_" + array_name)

            state.add_edge(
                state.add_read(array_name),
                None,
                map_entry,
                "IN_" + array_name,
                # f"IN_{conn_id}",
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]}*(tile_i+1),I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]}*(tile_j+1),J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                map_entry,
                "OUT_" + array_name,
                nsdfg_node,
                array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )
        # conn_id += 1
    # conn_id = 0
    for array_name, num_accesses in output_arrays.items():
        array = sdfg.arrays[array_name]

        if not array.transient:
            map_exit.add_in_connector("IN_" + array_name)
            map_exit.add_out_connector("OUT_" + array_name)
            state.add_edge(
                map_exit,
                "OUT_" + array_name,
                state.add_write(array_name),
                None,
                memlet=dace.Memlet.simple(
                    array_name,
                    subset_str=",".join(
                        [f"0:{limit}" if str(limit) != "1" else "0" for limit in array.shape]
                    ),
                    num_accesses=num_accesses,
                ),
            )
            from dace.data import Scalar

            if isinstance(array, dace.data.Scalar):
                subset_str = "0"
            else:
                frame_i = dace.symbolic.pystr_to_symbolic(str(array.shape[0]) + "-I")
                frame_j = dace.symbolic.pystr_to_symbolic(str(array.shape[1]) + "-J")
                subset_str = ",".join(
                    [
                        # f"{tile_size[0]}*tile_i:Min({tile_size[0]+1}*tile_i,I)+{frame_i}",
                        # f"{tile_size[1]}*tile_j:Min({tile_size[1]+1}*tile_j,J)+{frame_j}",
                        f"tile_i:Min(tile_i+{tile_size[0]},I)+{frame_i}",
                        f"tile_j:Min(tile_j+{tile_size[1]},J)+{frame_j}",
                        f"0:{array.shape[2]}",
                    ]
                )

            state.add_edge(
                nsdfg_node,
                array_name,
                map_exit,
                "IN_" + array_name,
                memlet=dace.Memlet.simple(
                    array_name, subset_str=subset_str, num_accesses=num_accesses
                ),
            )

    if len(input_arrays) == 0:
        state.add_edge(map_entry, None, nsdfg_node, None, dace.EmptyMemlet())
    if len(output_arrays) == 0:
        state.add_edge(nsdfg_node, None, map_exit, None, dace.EmptyMemlet())

    # dace.dtypes.StorageType.register("CPU_Threadprivate_Persistent")
    import sympy

    # symbols = dict(_tile_I=dace.symbol("_tile_I"), _tile_J=dace.symbol("_tile_J"))
    # symbols['_tile_I'].set(tile_size[0])
    # symbols['_tile_J'].set(tile_size[1])

    # tile_sizes = dict(I=tile_size[0], J=tile_size[1], K="K")
    for array_name, array in nsdfg_node.sdfg.arrays.items():
        if array.transient:
            # array.shape = [
            #     f"{tile_sizes[str(s)]}"
            #     if isinstance(s, dace.symbolic.symbol)
            #     else s.subs({a: tile_sizes[str(a)] for a in s.args if str(a) in "IJ"})
            #     for s in array.shape
            # ]
            array.tile_size = tile_size
            # print()
            array.storage = dace.dtypes.StorageType.CPU_ThreadLocal


import dace.transformation.pattern_matching as pattern_matching
from dace.properties import make_properties, Property, ShapeProperty
from dace import nodes
import dace.sdfg.utils


@registry.autoregister_params(singlestate=True)
@make_properties
class GlobalIJMapTiling(pattern_matching.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    prefix = Property(dtype=str, default="tile", desc="Prefix for new range symbols")
    tile_size = ShapeProperty(dtype=tuple, default=(8, 8), desc="Tile size per dimension")

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [dace.sdfg.utils.node_path_graph(GlobalIJMapTiling._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        # applies to IJ maps fully containing [0:I]x[0:J]
        map_entry = graph.nodes()[candidate[GlobalIJMapTiling._map_entry]]
        if not sorted(map_entry.map.params) == ["i", "j"]:
            return False
        free_symbols = map_entry.range.free_symbols
        if not sorted(free_symbols.keys()) == ["I", "J"]:
            return False
        I_symbol = free_symbols["I"]
        J_symbol = free_symbols["J"]
        range = map_entry.range.ranges
        if not range[0][0] <= 0 or not range[1][0] <= 0:
            return False
        if not range[0][1] - I_symbol <= 0 or not range[1][1] - J_symbol <= 0:
            return False
        if not range[0][2] == 1 or not range[1][2] == 1:
            return False

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[GlobalIJMapTiling._map_entry]]
        return map_entry.map.label + ": " + str(map_entry.map.params)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        map_entry = graph.nodes()[self.subgraph[GlobalIJMapTiling._map_entry]]

        # nd_to = symbolic.pystr_to_symbolic(
        #     f"int_ceil(I + 1, {tile_stride}) - 1"
        # )

        free_symbols = map_entry.range.free_symbols
        I_symbol = free_symbols["I"]
        J_symbol = free_symbols["J"]

        ranges = map_entry.range.ranges
        halo = (
            (ranges[0][0], ranges[0][1] - I_symbol + 1),
            (ranges[1][0], ranges[1][1] - J_symbol + 1),
        )

        from dace.transformation.dataflow.tiling import MapTiling

        maptiling = MapTiling(
            sdfg.sdfg_list.index(sdfg),
            self.state_id,
            {MapTiling._map_entry: self.subgraph[GlobalIJMapTiling._map_entry]},
            self.expr_index,
        )
        assert halo[0][0] <= 0
        assert halo[0][1] >= 0
        assert halo[1][0] <= 0
        assert halo[1][1] >= 0
        maptiling.tile_sizes = [t - h_l + h_r for t, (h_l, h_r) in zip(self.tile_size, halo)]
        maptiling.strides = self.tile_size
        maptiling.apply(sdfg)

        new_map_entry = graph.in_edges(map_entry)[0].src
        import dace.symbolic as symbolic

        new_map_entry.range.ranges[0] = (
            0,
            symbolic.pystr_to_symbolic(f"int_ceil(I, {self.tile_size[0]}) - 1"),
            1,
        )
        new_map_entry.range.ranges[1] = (
            0,
            symbolic.pystr_to_symbolic(f"int_ceil(J, {self.tile_size[1]}) - 1"),
            1,
        )


@registry.autoregister_params(singlestate=True)
@make_properties
class TaskletAsKLoop(pattern_matching.Transformation):
    """ Docstring TODO
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _tasklet = nodes.Tasklet("")
    _map_exit = nodes.MapExit(nodes.Map("", [], []))

    # Properties
    init = Property(default=0, desc="initial value for k")
    condition = Property(default="k<K", desc="stopping condition for the loop")
    step = Property(default="k+1", desc="value assigned to k every step (e.g. increment k+1)")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            dace.sdfg.utils.node_path_graph(
                TaskletAsKLoop._map_entry, TaskletAsKLoop._tasklet, TaskletAsKLoop._map_exit
            )
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    def _k_range(self):
        if "<" in self.condition:
            k_min = self.init
            _, k_max = self.condition.split("<")
            k_max = k_max + " - 1"
        else:
            k_max = str(self.init)
            _, k_min = self.condition.split(">=")
        return k_min, k_max

    def apply(self, sdfg):
        graph: dace.sdfg.SDFGState = sdfg.nodes()[self.state_id]
        map_entry: dace.nodes.MapEntry = graph.nodes()[self.subgraph[TaskletAsKLoop._map_entry]]
        tasklet: dace.nodes.Tasklet = graph.nodes()[self.subgraph[TaskletAsKLoop._tasklet]]
        map_exit: dace.nodes.MapExit = graph.nodes()[self.subgraph[TaskletAsKLoop._map_exit]]
        from dace.transformation.helpers import nest_state_subgraph

        k_min, k_max = self._k_range()
        # fix outer edges to ij map
        import sympy

        k_symbol = dace.symbolic.symbol("k")
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            for i, r in enumerate(e.data.subset.ranges):
                e.data.subset.ranges[i] = (
                    r[0].subs(dace.symbolic.symbol("k"), k_min),
                    r[1].subs(dace.symbolic.symbol("k"), k_max),
                    r[2],
                )

        # node = nest_state_subgraph(sdfg, graph, dace.sdfg.ScopeSubgraphView(graph, [tasklet]))
        nsdfg: SDFG = dace.SDFG(f"nested_k_loop_{graph.name}")
        nstate = nsdfg.add_state()
        nstate.add_nodes_from([tasklet])
        # nsdfg.add_nodes_from(dace.sdfg.ScopeSubgraphView(graph, [nstate]))

        in_prefix = f"__in_"
        out_prefix = f"__out_"

        nsdfg_in_arrays = set()
        for e in graph.out_edges(map_entry):
            nsdfg_in_arrays.add(in_prefix + e.data.data)
        nsdfg_out_arrays = set()
        for e in graph.in_edges(map_exit):
            nsdfg_out_arrays.add(out_prefix + e.data.data)

        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, dace.nodes.AccessNode) and n.access == dace.dtypes.AccessType.ReadOnly
        ):
            nsdfg.add_datadesc(in_prefix + name, sdfg.arrays[name])
        for name in set(
            n.data
            for n in graph.nodes()
            if isinstance(n, dace.nodes.AccessNode)
            and n.access == dace.dtypes.AccessType.WriteOnly
        ):
            nsdfg.add_datadesc(out_prefix + name, sdfg.arrays[name])

        read_accessors = dict()
        for name in nsdfg_in_arrays:
            read_accessors[name] = nstate.add_read(name)
        write_accessors = dict()
        for name in nsdfg_out_arrays:
            write_accessors[name] = nstate.add_write(name)

        for e in graph.out_edges(map_entry):
            nstate.add_edge(
                read_accessors[in_prefix + e.data.data],
                None,
                tasklet,
                e.dst_conn,
                memlet=dace.Memlet.simple(
                    in_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )
        for e in graph.in_edges(map_exit):
            nstate.add_edge(
                tasklet,
                e.src_conn,
                write_accessors[out_prefix + e.data.data],
                None,
                memlet=dace.Memlet.simple(
                    out_prefix + e.data.data,
                    subset_str=str(e.data.subset),
                    num_accesses=e.data.num_accesses,
                ),
            )

        node = graph.add_nested_sdfg(nsdfg, sdfg, nsdfg_in_arrays, nsdfg_out_arrays)
        nstate = nsdfg.nodes()[0]

        conn_map_entry_to_nsdfg = dict()
        subsets_map_entry_to_nsdfg = dict()
        num_map_entry_to_nsdfg = dict()
        for e in graph.out_edges(map_entry):
            conn_map_entry_to_nsdfg[e.src_conn] = e.data.data

            subset = subsets_map_entry_to_nsdfg.get(e.data.data, e.data.subset)
            num = num_map_entry_to_nsdfg.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, dace.nodes.AccessNode)
                    #         and n.access == dace.AccessType.ReadOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    #     map_entry,
                    # )[0].data.subset.ranges[i]
                subsets_map_entry_to_nsdfg[e.data.data] = subset
                num_map_entry_to_nsdfg[e.data.data] = num + e.data.num_accesses

        conn_map_exit_to_nsdfg = dict()
        for e in graph.in_edges(map_exit):
            conn_map_exit_to_nsdfg[e.dst_conn] = e.data.data

        for conn in map_entry.out_connectors:
            data_name = conn_map_entry_to_nsdfg[conn]
            graph.add_edge(
                map_entry,
                conn,
                node,
                in_prefix + conn_map_entry_to_nsdfg[conn],
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_map_entry_to_nsdfg[data_name]),
                    num_accesses=num_map_entry_to_nsdfg[data_name],
                ),
            )

        conn_nsdfg_to_map_exit = dict()
        subsets_nsdfg_to_map_exit = dict()
        num_nsdfg_to_map_exit = dict()
        for e in graph.in_edges(map_exit):
            conn_nsdfg_to_map_exit[e.dst_conn] = e.data.data

            subset = subsets_nsdfg_to_map_exit.get(e.data.data, e.data.subset)
            num = num_nsdfg_to_map_exit.get(e.data.data, e.data.num_accesses)
            for i, r in enumerate(subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        min(subset.ranges[i][0], e.data.subset[i][0]),
                        max(subset.ranges[i][1], e.data.subset[i][1]),
                        1,
                    )
                elif "k" in dace.symbolic.symlist(r):
                    subset.ranges[i] = (
                        0,
                        dace.symbolic.pystr_to_symbolic("K-1"),
                        1,
                    )  # graph.edges_between(
                    #     map_exit,
                    #     [
                    #         n
                    #         for n in graph.nodes()
                    #         if isinstance(n, dace.nodes.AccessNode)
                    #         and n.access == dace.AccessType.WriteOnly
                    #         and n.data == e.data.data
                    #     ][0],
                    # )[0].data.subset.ranges[i]
                subsets_nsdfg_to_map_exit[e.data.data] = subset
                num_nsdfg_to_map_exit[e.data.data] = num + e.data.num_accesses
        for conn in map_exit.in_connectors:
            data_name = conn_nsdfg_to_map_exit[conn]
            graph.add_edge(
                node,
                out_prefix + conn_map_exit_to_nsdfg[conn],
                map_exit,
                conn,
                memlet=dace.Memlet.simple(
                    data=data_name,
                    subset_str=str(subsets_nsdfg_to_map_exit[data_name]),
                    num_accesses=num_nsdfg_to_map_exit[data_name],
                ),
            )
        for e in graph.in_edges(map_entry) + graph.out_edges(map_exit):
            if len(e.data.subset.ranges) >= 3 and "k" in dace.symbolic.symlist(
                e.data.subset.ranges[2]
            ):
                e.data.subset.ranges[2] = (0, dace.symbolic.pystr_to_symbolic("K-1"), 1)

        for e in nstate.in_edges(tasklet):
            outer_subset = subsets_map_entry_to_nsdfg[e.data.data[len(in_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        for e in nstate.out_edges(tasklet):
            outer_subset = subsets_nsdfg_to_map_exit[e.data.data[len(out_prefix) :]]
            for i, r in enumerate(e.data.subset.ranges):
                if "i" in dace.symbolic.symlist(r) or "j" in dace.symbolic.symlist(r):
                    e.data.subset.ranges[i] = (
                        r[0] - outer_subset.ranges[i][0],
                        r[1] - outer_subset.ranges[i][0],
                        1,
                    )

        # Create a loop inside the nested SDFG
        nsdfg.add_loop(None, nstate, None, "k", self.init, self.condition, self.step)
        graph.remove_node(tasklet)
        # outer_in_edges = {e.dst_conn: e for e in graph.in_edges(node)}
        # outer_out_edges = {e.src_conn: e for e in graph.out_edges(node)}
        #
        # for e in nstate.in_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.src.data in outer_in_edges
        #     outer_edge = outer_in_edges[e.src.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r
        #
        # for e in nstate.out_edges(tasklet):
        #     assert all(r == (0, 0, 1) for r in e.data.subset.ranges)
        #     assert e.dst.data in outer_out_edges
        #     outer_edge = outer_out_edges[e.dst.data]
        #     for i, r in enumerate(outer_edge.data.subset.ranges):
        #         e.data.subset.ranges[i] = r

        #     e.data.subset.ranges[i] = r
        # if len(e.data.subset.ranges) > 2:
        #     e.data.subset.ranges[2] = (
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("k"),
        #         dace.symbolic.pystr_to_symbolic("1"),
        #     )


def eliminate_trivial_k_loop(sdfg: dace.SDFG, state: dace.SDFGState):
    sdfg.predecessor_states(state)
    if not len(sdfg.successors(state)) == 2:
        return
    if not len(sdfg.predecessors(state)) == 2:
        return
    init, condition, step = None, None, None
    for s in sdfg.predecessors(state):
        edges = sdfg.edges_between(s, state)
        if not len(edges) == 1:
            return
        if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
            init = edges[0].data.assignments["k"]
            init_state = s
        elif not edges[0].data.condition.as_string == "":
            return
        else:
            step = edges[0].data.assignments["k"]
            loop_end_state = s
    for s in sdfg.successors(state):
        edges = sdfg.edges_between(state, s)
        if edges:
            if not len(edges) == 1:
                return
            if not edges[0].data.condition.as_string == "":
                condition = edges[0].data.condition
                loop_start_state = s
            else:
                exit_state = s

    if "<" in condition.as_string:
        k_min = init
        _, k_max = condition.as_string.split("<")
        k_max = k_max + " - 1"
    else:
        k_max = str(init)
        _, k_min = condition.as_string.split(">=")

    if not dace.symbolic.pystr_to_symbolic(f"({k_min})-({k_max})") == 0:
        return

    # add edge from pred directly to loop states
    sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge(assignments={"k": init}))
    # add edge from loop states directly to succ
    sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
    # remove guard & edges involving guard
    for s in sdfg.successors(state):
        for edge in sdfg.edges_between(state, s):
            sdfg.remove_edge(edge)
    for s in sdfg.predecessors(state):
        for edge in sdfg.edges_between(s, state):
            sdfg.remove_edge(edge)
    sdfg.remove_node(state)


def outer_k_loop_to_inner_map(sdfg: dace.SDFG, state: dace.SDFGState):
    sdfg.predecessor_states(state)
    if not len(sdfg.successors(state)) == 2:
        return
    if not len(sdfg.predecessors(state)) == 2:
        return
    init, condition, step = None, None, None
    for s in sdfg.predecessors(state):
        edges = sdfg.edges_between(s, state)
        if not len(edges) == 1:
            return
        if edges[0].data.condition.as_string == "" and s in sdfg.predecessor_states(state):
            init = edges[0].data.assignments["k"]
            init_state = s
        elif not edges[0].data.condition.as_string == "":
            return
        else:
            step = edges[0].data.assignments["k"]
            loop_end_state = s
    for s in sdfg.successors(state):
        edges = sdfg.edges_between(state, s)
        if edges:
            if not len(edges) == 1:
                return
            if not edges[0].data.condition.as_string == "":
                condition = edges[0].data.condition
                loop_start_state = s
            else:
                exit_state = s
    # for state in loop...
    loop_states = []
    s = loop_start_state
    while s is not state:
        if not len(sdfg.successors(s)) == 1:
            return
        else:
            loop_states.append(s)
            s = sdfg.successors(s)[0]
    assert loop_end_state is loop_states[-1]

    # replace tasklet with nestedsdfg
    for s in loop_states:
        sdfg.apply_transformations(
            TaskletAsKLoop,
            states=[s],
            validate=False,
            options=dict(init=init, step=step, condition=condition.as_string),
        )
    # add edge from pred directly to loop states
    sdfg.add_edge(init_state, loop_start_state, dace.InterstateEdge())
    # add edge from loop states directly to succ
    sdfg.add_edge(loop_end_state, exit_state, dace.InterstateEdge())
    # remove guard & edges involving guard
    for s in sdfg.successors(state):
        for edge in sdfg.edges_between(state, s):
            sdfg.remove_edge(edge)
    for s in sdfg.predecessors(state):
        for edge in sdfg.edges_between(s, state):
            sdfg.remove_edge(edge)
    sdfg.remove_node(state)
