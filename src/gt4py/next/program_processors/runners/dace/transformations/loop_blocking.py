# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Optional, Union

import dace
from dace import (
    data as dace_data,
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import graph as dace_graph, nodes as dace_nodes, propagation as dace_propagation
from dace.transformation import helpers as dace_helpers

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import gtir_to_sdfg_utils


@dace_properties.make_properties
class LoopBlocking(dace_transformation.SingleStateTransformation):
    """Applies loop blocking, also known as k-blocking, on a Map.

    This transformation takes a multidimensional Map and performs blocking on a
    single dimension, the loop variable is called `I` here. All dimensions except
    `I` are unaffected by this transformation. In the outer Map will be replace the
    `I` range, currently `I = 0:N`, with `__coarse_I = 0:N:B`, where `N` is the
    original size of the range and `B` is the blocking size. The transformation
    will then create an inner sequential map with `I = __coarse_I:(__coarse_I + B)`.

    What makes this transformation different from simple blocking, is that
    the inner map will not just be inserted right after the outer Map.
    Instead the transformation will first identify all nodes that does not depend
    on the blocking parameter `I`, called independent nodes and relocate them
    between the outer and inner map. Note that an independent node must be connected
    to the MapEntry or another independent node.
    Thus these operations will only be performed once, per outer loop iteration.

    Args:
        blocking_size: The size of the block, denoted as `B` above.
        blocking_parameter: On which parameter should we block.
        require_independent_nodes: If `True` only apply loop blocking if the Map
            actually contains independent nodes. Defaults to `True`.

    Todo:
        - Modify the inner map such that it always starts at zero.
        - Allow more than one parameter on which we block.
    """

    blocking_size = dace_properties.Property(
        dtype=int,
        allow_none=True,
        desc="Size of the inner blocks; 'B' in the above description.",
    )
    blocking_parameter = dace_properties.Property(
        dtype=str,
        allow_none=True,
        desc="Name of the iteration variable on which to block (must be an exact match);"
        " 'I' in the above description.",
    )
    require_independent_nodes = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="If 'True' then blocking is only applied if there are independent nodes.",
    )

    # Set of nodes that are independent of the blocking parameter.
    _independent_nodes: Optional[set[dace_nodes.AccessNode]]
    _dependent_nodes: Optional[set[dace_nodes.AccessNode]]

    outer_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        blocking_size: Optional[int] = None,
        blocking_parameter: Optional[Union[gtx_common.Dimension, str]] = None,
        require_independent_nodes: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if isinstance(blocking_parameter, gtx_common.Dimension):
            blocking_parameter = gtir_to_sdfg_utils.get_map_variable(blocking_parameter)
        if blocking_parameter is not None:
            self.blocking_parameter = blocking_parameter
        if blocking_size is not None:
            self.blocking_size = blocking_size
        if require_independent_nodes is not None:
            self.require_independent_nodes = require_independent_nodes
        self._independent_nodes = None
        self._dependent_nodes = None

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.outer_entry)]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Test if the map can be blocked.

        The test involves:
        - The map must be at the top level.
        - The block dimension must be present (exact string match).
        - The map range must have step size of 1.
        - The partition must exists (see `partition_map_output()`).
        """
        if self.blocking_parameter is None:
            raise ValueError("The blocking dimension was not specified.")
        elif self.blocking_size is None:
            raise ValueError("The blocking size was not specified.")

        outer_entry: dace_nodes.MapEntry = self.outer_entry
        map_params: list[str] = outer_entry.map.params
        map_range: dace_subsets.Range = outer_entry.map.range
        block_var: str = self.blocking_parameter

        scope = graph.scope_dict()
        if scope[outer_entry] is not None:
            return False

        if block_var not in map_params:
            return False

        block_var_idx = map_params.index(block_var)
        map_range_size = map_range.size()
        if map_range[block_var_idx][2] != 1:
            return False

        # Require that there are more iteration than the blocking size.
        # TODO(phimuell): Synchronize this with the GPU block size since it also
        #   plays into it.
        if (map_range_size[block_var_idx] <= self.blocking_size) == True:  # noqa: E712 [true-false-comparison]  # SymPy Fancy comparison.
            return False

        if not self.partition_map_output(graph, sdfg):
            return False
        self._independent_nodes = None
        self._dependent_nodes = None

        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        """Creates a blocking map.

        Performs the operation described in the doc string.
        """
        # Now compute the partitions of the nodes.
        self.partition_map_output(graph, sdfg)

        # Modify the outer map and create the inner map.
        (outer_entry, outer_exit), (inner_entry, inner_exit) = self._prepare_inner_outer_maps(graph)

        # Reconnect the edges
        self._rewire_map_scope(
            outer_entry=outer_entry,
            outer_exit=outer_exit,
            inner_entry=inner_entry,
            inner_exit=inner_exit,
            state=graph,
            sdfg=sdfg,
        )
        self._independent_nodes = None
        self._dependent_nodes = None

    def _prepare_inner_outer_maps(
        self,
        state: dace.SDFGState,
    ) -> tuple[
        tuple[dace_nodes.MapEntry, dace_nodes.MapExit],
        tuple[dace_nodes.MapEntry, dace_nodes.MapExit],
    ]:
        """Prepare the maps for the blocking.

        The function modifies the outer map, `self.outer_entry`, by replacing the
        blocking parameter, `self.blocking_parameter`, with a coarsened version
        of it. In addition the function will then create the inner map, that
        iterates over the blocking parameter, and these bounds are determined
        by the coarsened blocking parameter of the outer map.

        Args:
            state: The state on which we operate.

        Return:
            The function returns a tuple of length two, the first element is the
            modified outer map and the second element is the newly created
            inner map. Each element consist of a pair containing the map entry
            and map exit nodes of the corresponding maps.
        """
        outer_entry: dace_nodes.MapEntry = self.outer_entry
        outer_exit: dace_nodes.MapExit = state.exit_node(outer_entry)
        outer_map: dace_nodes.Map = outer_entry.map
        outer_range: dace_subsets.Range = outer_entry.map.range
        outer_params: list[str] = outer_entry.map.params
        blocking_parameter_dim = outer_params.index(self.blocking_parameter)

        # This is the name of the iterator that we use in the outer map for the
        #  blocked dimension
        # NOTE: The way we create the new parameter name is important such that it
        #   could be matched through the `unit_strides_kind` argument, which is
        #   the case with this approach. But it makes the `unit_strides_dim`
        #   argument `gt_set_iteration_order()` inapplicable.
        coarse_block_var = "__gtx_coarse_" + self.blocking_parameter

        # Generate the sequential inner map
        rng_start = outer_range[blocking_parameter_dim][0]
        rng_stop = outer_range[blocking_parameter_dim][1]
        inner_label = f"inner_{outer_map.label}"
        inner_range = {
            self.blocking_parameter: dace_subsets.Range.from_string(
                f"(({rng_start}) + ({coarse_block_var}) * ({self.blocking_size})):"
                f"min(({rng_start}) + ({coarse_block_var} + 1) * ({self.blocking_size}), ({rng_stop}) + 1)"
            )
        }
        inner_entry, inner_exit = state.add_map(
            name=inner_label,
            ndrange=inner_range,
            schedule=dace.dtypes.ScheduleType.Sequential,
            debuginfo=copy.copy(outer_map.debuginfo),
        )

        # TODO(phimuell): Investigate if we want to prevent unrolling here

        # Now we modify the properties of the outer map.
        coarse_block_range = dace_subsets.Range.from_string(
            f"0:int_ceil((({rng_stop}) + 1) - ({rng_start}), ({self.blocking_size}))"
        ).ranges[0]
        outer_map.params[blocking_parameter_dim] = coarse_block_var
        outer_map.range[blocking_parameter_dim] = coarse_block_range

        return ((outer_entry, outer_exit), (inner_entry, inner_exit))

    def partition_map_output(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """Computes the partition the of the nodes of the Map.

        The function divides the nodes into two sets, defined as:
        - The independent nodes `\mathcal{I}`:
            These are the nodes, whose output does not depend on the blocked
            dimension. These nodes can be relocated between the outer and inner map.
            Nodes in these set does not necessarily have a direct edge to map entry.
            However, the exists a path from `outer_entry` to any node in this set.
        - The dependent nodes `\mathcal{D}`:
            These are the nodes, whose output depend on the blocked dimension.
            Thus they can not be relocated between the two maps, but will remain
            inside the inner scope. All these nodes have at least one edge to map entry.

        The function uses an iterative method to compute the set of independent nodes.
        In each iteration the function will look classify all nodes that have an
        incoming edge originating either at outer_entry or from a node that was
        already classified as independent. This is repeated until no new independent
        nodes are found. This means that independent nodes does not necessarily have
        a direct connection to map entry.

        The dependent nodes on the other side always have a direct edge to outer_entry.
        As they are the set of nodes that are adjacent to outer_entry but are not
        independent.

        For the sake of arguments, all nodes, except the map entry and exit nodes,
        that are inside the scope and are not classified as dependent or independent
        are known as "inner nodes".

        If the function is able to compute the partition `True` is returned and the
        member variables are updated. If the partition does not exists the function
        will return `False` and the respective member variables will be `None`.

        The function will honor `self.require_independent_nodes`. Thus if no independent
        nodes were found the function behaves as if the partition does not exist.

        Args:
            state: The state on which we operate.
            sdfg: The SDFG in which we operate on.

        Note:
            - The function will only inspect the direct children of the map entry.
            - Currently this function only considers the input Memlets and the
                `used_symbol` properties of a Tasklet to determine if a Tasklet is dependent.
        """

        # Clear the previous partition.
        self._independent_nodes = set()
        self._dependent_nodes = None

        while True:
            # Find all the nodes that we have to classify in this iteration.
            #  - All nodes adjacent to `outer_entry` (which is
            #       independent by definition).
            #  - All nodes adjacent to independent nodes.
            nodes_to_classify: set[dace_nodes.Node] = {
                edge.dst for edge in state.out_edges(self.outer_entry)
            }
            for independent_node in self._independent_nodes:
                nodes_to_classify.update({edge.dst for edge in state.out_edges(independent_node)})
            nodes_to_classify.difference_update(self._independent_nodes)

            # Now classify each node
            found_new_independent_node = False
            for node_to_classify in nodes_to_classify:
                class_res = self._classify_node(
                    node_to_classify=node_to_classify,
                    state=state,
                    sdfg=sdfg,
                )

                # Check if the partition exists.
                if class_res is None:
                    self._independent_nodes = None
                    return False
                if class_res is True:
                    found_new_independent_node = True

            # If we found a new independent node then we have to continue.
            if not found_new_independent_node:
                break

        # Perform some cleaning on the independent nodes.
        self._post_process_independent_nodes(state)

        assert all(
            all(
                iedge.src in self._independent_nodes or iedge.src is self.outer_entry
                for iedge in state.in_edges(inode)
            )
            for inode in self._independent_nodes
        )

        # If requested check if the blocking is a good idea.
        if self.require_independent_nodes and (not self._check_if_blocking_is_favourable(state)):
            self._independent_nodes = None
            return False

        # After the independent set is computed compute the set of dependent nodes
        #  as the set of all nodes adjacent to `outer_entry` that are not independent.
        self._dependent_nodes = {
            edge.dst
            for edge in state.out_edges(self.outer_entry)
            if edge.dst not in self._independent_nodes
        }

        return True

    def _classify_node(
        self,
        node_to_classify: dace_nodes.Node,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool | None:
        """Internal function for classifying a single node.

        The general rule to classify if a node is independent are:
        - The node must be a Tasklet or an AccessNode, in all other cases the
            partition does not exist.
        - `free_symbols` of the nodes shall not contain the blocking parameter.
        - All incoming _empty_ edges must be connected to the map entry.
        - A node either has only empty Memlets or none of them.
        - Incoming Memlets does not depend on the blocking parameter.
        - All incoming edges must start either at `self.outer_entry` or at dependent nodes.
        - All output Memlets are non empty.

        It is important that to realize that the function will add the node to
        classify to `self._independent_nodes` on its own. It is also important that
        the function might add other nodes beside `node_to_classify`.

        Returns:
            The function returns `True` if `node_to_classify` is considered independent.
            If the function returns `False` the node was classified as a dependent node.
            The function will return `None` if the node can not be classified, in this
            case the partition does not exist.

        Args:
            node_to_classify: The node that should be classified.
            state: The state containing the map.
            sdfg: The SDFG that is processed.
        """
        assert self._independent_nodes is not None  # silence MyPy
        outer_entry: dace_nodes.MapEntry = self.outer_entry  # for caching.
        outer_exit: dace_nodes.MapExit = state.exit_node(outer_entry)

        # The node needs to have an input and output.
        if state.in_degree(node_to_classify) == 0 or state.out_degree(node_to_classify) == 0:
            return None

        # The outer MapExit is always classified as dependent.
        if node_to_classify is outer_exit:
            return False

        # To fully understand what is going on we have to look at the input and output
        #  edges of the node. We define them here to allow to modify them in certain
        #  cases.
        in_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = list(
            state.in_edges(node_to_classify)
        )
        out_edges: list[dace_graph.MultiConnectorEdge[dace.Memlet]] = list(
            state.out_edges(node_to_classify)
        )

        # Despite its type the node's free symbols can not contain the blocking
        #  parameter. In case of a Tasklet this would be the body of the Tasklet.
        if self.blocking_parameter in node_to_classify.free_symbols:
            return False

        # If the test succeed then these are the nodes we additionally consider
        #  as independent.
        new_independent_nodes: set[dace_nodes.Node] = {node_to_classify}

        # We are only able to handle certain kind of nodes, so screening them.
        if isinstance(node_to_classify, dace_nodes.Tasklet):
            if node_to_classify.side_effects:
                return None

            # A Tasklet must write to an AccessNode, because otherwise there would
            #  be nothing that could be used to cache anything. Furthermore, this
            #  AccessNode must be outside of the inner loop, i.e. be independent.
            # TODO: Make this check stronger to ensure that there is always an
            #   AccessNode that is independent.
            if not all(
                isinstance(out_edge.dst, dace_nodes.AccessNode)
                for out_edge in state.out_edges(node_to_classify)
                if not out_edge.data.is_empty()
            ):
                return False

        elif isinstance(node_to_classify, dace.nodes.NestedSDFG):
            # Same check as for Tasklets applies to the outputs of a nested SDFG node
            if not all(
                isinstance(out_edge.dst, dace_nodes.AccessNode)
                for out_edge in state.out_edges(node_to_classify)
                if not out_edge.data.is_empty()
            ):
                return False

            # Additionally, test if the symbol mapping depends on the block variable.
            for v in node_to_classify.symbol_mapping.values():
                if self.blocking_parameter in v.free_symbols:
                    return False

        elif isinstance(node_to_classify, dace_nodes.AccessNode):
            # AccessNodes need to have some special properties.
            node_desc: dace.data.Data = node_to_classify.desc(sdfg)
            if isinstance(node_desc, dace.data.View):
                # Views are forbidden.
                return None

            # The access node inside either has scope lifetime or is a scalar.
            if isinstance(node_desc, dace.data.Scalar):
                pass
            elif node_desc.lifetime != dace.dtypes.AllocationLifetime.Scope:
                return None

        elif isinstance(node_to_classify, dace_nodes.MapEntry):
            # We check a Map as a whole and add it as a whole to the set of the
            #  independent nodes. For that reason we must modify `out_edges` since
            #  now the outputs of the associated `MapExit` must be checked.
            #  However, we have to run some Map specific checks.
            # TODO(phimuell): Do we also have to modify `node_to_classify`?
            map_exit = state.exit_node(node_to_classify)

            # The blocking parameter can not be used inside the Map scope.
            map_scope = state.scope_subgraph(node_to_classify)
            if self.blocking_parameter in map_scope.free_symbols:
                return False

            # There is an obscure case, where the Memlet on the inside of a Map scope
            #  and the Memlet on the outside refer to different data, thus we have to
            #  check that here. Note that we only have to do it here in this case
            #  because normally it would be spotted above where we checked the input.
            out_edges = list(state.out_edges(map_exit))
            if any(self.blocking_parameter in out_edge.data.free_symbols for out_edge in out_edges):
                return False

            # Add all nodes of the Map scope, including entry and exit node to the
            #  set of new independent nodes.
            new_independent_nodes.update(map_scope.nodes())

        elif isinstance(node_to_classify, dace.libraries.standard.nodes.Reduce):
            # The only checks we impose on them is the free symbols check and the
            #  input output checks.
            pass

        else:
            # Any other node type we can not handle, so the partition can not exist.
            # TODO(phimuell): Try to handle certain kind of library nodes.
            return None

        # In a first phase we will only look if the partition exists or not.
        #  We will therefore not check if the node is independent or not, since
        #  for these classification to make sense the partition has to exist in the
        #  first place.

        # There are some very small requirements that we impose on the output edges.
        for out_edge in out_edges:
            # We consider nodes that are directly connected to the outer map exit as
            #  dependent. This is an implementation detail to avoid some hard cases.
            #  The only exceptions are scalars which we will handle later.
            # NOTE: We restrict ourself to scalars to make sure that no dynamic
            #  allocation is needed. It could be extended to arrays later.
            if out_edge.dst is outer_exit:
                if (
                    isinstance(node_to_classify, dace_nodes.AccessNode)
                    and isinstance(node_to_classify.desc(sdfg), dace_data.Scalar)
                    and out_edge.data.wcr is None
                ):
                    continue
                return False

        # Now we have to look at incoming edges individually.
        #  We will inspect the subset of the Memlet to see if they depend on the
        #  block variable. If this loop ends normally, then we classify the node
        #  as independent and the node is added to `_independent_nodes`.
        for in_edge in in_edges:
            memlet: dace.Memlet = in_edge.data
            src_subset: dace_subsets.Subset | None = memlet.src_subset
            dst_subset: dace_subsets.Subset | None = memlet.dst_subset

            if memlet.is_empty():  # Empty Memlets do not impose any restrictions.
                continue

            # Now we have to look at the source and destination set of the Memlet.
            subsets_to_inspect: list[dace_subsets.Subset] = []
            if dst_subset is not None:
                subsets_to_inspect.append(dst_subset)
            if src_subset is not None:
                subsets_to_inspect.append(src_subset)

            # If a subset needs the block variable then the node is not independent
            #  but dependent.
            if any(self.blocking_parameter in subset.free_symbols for subset in subsets_to_inspect):
                return False

            # The edge must either originate from `outer_entry` or from an independent
            #  node if not it is dependent.
            if not (in_edge.src is outer_entry or in_edge.src in self._independent_nodes):
                return False

        # Loop ended normally, thus we update the list of independent nodes.
        self._independent_nodes.update(new_independent_nodes)
        return True

    def _post_process_independent_nodes(
        self,
        state: dace.SDFGState,
    ) -> None:
        """Cleans the set of independent nodes.

        This function is mostly there to handle cases that are not implemented.
        This function might remove nodes from the set of independent nodes.
        """
        assert self._independent_nodes is not None  # silence MyPy

        independent_nodes_were_updated = True
        while independent_nodes_were_updated:
            independent_nodes_were_updated = False
            for node in list(self._independent_nodes):
                # The only nodes that are important here, are the independent nodes
                #  at the boundaries, i.e. independent nodes that are connected to
                #  dependent nodes.
                if all(
                    oedge.dst in self._independent_nodes
                    for oedge in state.out_edges(node)
                    if not oedge.data.is_empty()
                ):
                    continue

                if isinstance(node, dace_nodes.AccessNode):
                    # This is actually the only case that is implemented, which is also the
                    #  only case that makes sense, as the AccessNode is needed as cache.
                    pass

                elif isinstance(node, dace_nodes.Tasklet):
                    # A Tasklet "generates" some data that has to be stored somewhere, for
                    #  example in an AccessNode, that has to be independent. Thus we will
                    #  now remove the node from the set of independent nodes.
                    self._independent_nodes.remove(node)
                    independent_nodes_were_updated = True
                    break

                else:
                    # We can not handle this kind of boundary node, so remove it.
                    self._independent_nodes.remove(node)
                    independent_nodes_were_updated = True
                    break

    def _rewire_map_scope(
        self,
        outer_entry: dace_nodes.MapEntry,
        outer_exit: dace_nodes.MapExit,
        inner_entry: dace_nodes.MapEntry,
        inner_exit: dace_nodes.MapExit,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        """Rewire the edges inside the scope defined by the outer map.

        The function assumes that the outer and inner map were obtained by a call
        to `_prepare_inner_outer_maps()`. The function will now rewire the connections of these
        nodes such that the dependent nodes are inside the scope of the inner map,
        while the independent nodes remain outside.

        Args:
            outer_entry: The entry node of the outer map.
            outer_exit: The exit node of the outer map.
            inner_entry: The entry node of the inner map.
            inner_exit: The exit node of the inner map.
            state: The state of the map.
            sdfg: The SDFG we operate on.
        """
        assert self._independent_nodes is not None and self._dependent_nodes is not None

        # Contains the nodes that are already have been handled.
        relocated_nodes: set[dace_nodes.Node] = set()

        # We now handle all independent nodes, this means that all of their
        #  _output_ edges have to go through the new inner map and the Memlets
        #  need modifications, because of the block parameter.
        for independent_node in self._independent_nodes:
            for out_edge in list(state.out_edges(independent_node)):
                edge_dst: dace_nodes.Node = out_edge.dst
                relocated_nodes.add(edge_dst)

                # If destination of this edge is also independent we do not need
                #  to handle it, because that node will also be before the new
                #  inner serial map.
                if edge_dst in self._independent_nodes:
                    continue

                # Special case, when we encounter an AccessNode, referring to a scalar,
                #  that is connected to the outer MapExit node, then we will insert
                #  a copy Tasklet, thus the copy Tasklet will then serve as dependent
                #  node that writes into the output.
                if isinstance(independent_node, dace_nodes.AccessNode) and edge_dst is outer_exit:
                    assert isinstance(independent_node.desc(sdfg), dace_data.Scalar)
                    assert not out_edge.data.is_empty()

                    copy_tlet = state.add_tasklet(
                        name=f"loop_blocking_copy_tlet_{independent_node.data}_{id(out_edge)}",
                        inputs={"__in"},
                        outputs={"__out"},
                        code="__out = __in",
                    )

                    # remove the current output edge from the state, but we need it.
                    org_out_edge = out_edge
                    state.remove_edge(org_out_edge)

                    # Now create a new `out_node`, that connects the independent node
                    #  with the copy Tasklet. We can use a plain Memlet for that.
                    #  This is the edge that we will split later.
                    out_edge = state.add_edge(
                        independent_node,
                        org_out_edge.src_conn,
                        copy_tlet,
                        "__in",
                        dace.Memlet(f"{independent_node.data}[0]"),
                    )

                    # Create the edge that connects the copy Tasklet with the outer MapExit.
                    #  Here we are copying the original Memlet, however, because the
                    #  new source is a Tasklet, instead of an AccessNode we must clear
                    #  the source subset.
                    new_output_edge = state.add_edge(
                        copy_tlet,
                        "__out",
                        outer_exit,
                        org_out_edge.dst_conn,
                        dace.Memlet.from_memlet(org_out_edge.data),
                    )
                    new_output_edge.data.src_subset = None
                    assert new_output_edge.data.dst_subset is not None

                    # Update `edge_dst` and mark the copy Tasklet as processed.
                    edge_dst = copy_tlet
                    relocated_nodes.add(copy_tlet)

                # Now split `out_edge` such that it passes through the new inner entry.
                #  We do not need to modify the subsets, i.e. replacing the variable
                #  on which we block, because the node is independent and the outgoing
                #  new inner map entry iterate over the blocked variable.
                if out_edge.data.is_empty():
                    # `out_edge` is an empty Memlet that ensures its source, which is
                    #  independent, is sequenced before its destination, which is
                    #  dependent. We now have to split the Memlet into two.
                    # TODO(phimuell): Can we remove this edge? Is the map enough to
                    #   ensure proper sequencing?
                    new_in_conn = None
                    new_out_conn = None
                    new_memlet_outside = dace.Memlet()

                elif not isinstance(independent_node, dace_nodes.AccessNode):
                    # For syntactical reasons the boundary, i.e. independent nodes
                    #  that have a connection to a dependent node, must be AccessNodes,
                    #  because something is needed as cache. Thus if you hit this
                    #  case then there is a bug in the `classify_node()` function
                    #  or your SDFG is wrong.
                    # NOTE: We do not allow direct connections between Tasklets.
                    raise NotImplementedError()

                else:
                    # NOTE: This creates more connections that are ultimately
                    #  necessary. However, figuring out which one to use and if
                    #  it would be valid, is very complicated, so we don't do it.
                    new_map_conn = inner_entry.next_connector(try_name=out_edge.data.data)
                    new_in_conn = "IN_" + new_map_conn
                    new_out_conn = "OUT_" + new_map_conn
                    new_memlet_outside = dace.Memlet.from_array(
                        out_edge.data.data, sdfg.arrays[out_edge.data.data]
                    )
                    inner_entry.add_in_connector(new_in_conn)
                    inner_entry.add_out_connector(new_out_conn)

                state.add_edge(
                    out_edge.src,
                    out_edge.src_conn,
                    inner_entry,
                    new_in_conn,
                    new_memlet_outside,
                )
                state.add_edge(
                    inner_entry,
                    new_out_conn,
                    out_edge.dst,
                    out_edge.dst_conn,
                    copy.deepcopy(out_edge.data),
                )
                state.remove_edge(out_edge)

        # Now we handle the dependent nodes, they differ from the independent nodes
        #  in that they _after_ the new inner map entry. Thus, we have to modify
        #  their incoming edges.
        for dependent_node in self._dependent_nodes:
            for in_edge in state.in_edges(dependent_node):
                edge_src: dace_nodes.Node = in_edge.src

                # The incoming edge of a dependent node (before any processing) either
                #  starts at:
                #   - The outer map.
                #   - An other dependent node.
                #   - An independent node.
                #  The last case was already handled by the loop above.
                if edge_src is inner_entry:
                    # Edge originated originally at an independent node, but was
                    #  already handled by the loop above.
                    assert dependent_node in relocated_nodes

                elif edge_src is not outer_entry:
                    # Edge originated at an other dependent node. There is nothing
                    #  that we have to do.
                    # NOTE: We can not test if `edge_src` is in `self._dependent_nodes`
                    #  because it only contains the dependent nodes that are directly
                    #  connected to the map entry.
                    assert edge_src not in self._independent_nodes

                elif in_edge.data.is_empty():
                    # The dependent node has an empty Memlet to the other map.
                    #  Since the inner map is sequenced after the outer map,
                    #  we will simply reconnect the edge to the inner map.
                    # TODO(phimuell): Are there situations where this makes problems.
                    dace_helpers.redirect_edge(state=state, edge=in_edge, new_src=inner_entry)

                elif edge_src is outer_entry:
                    # This dependent node originated at the outer map. Thus we have to
                    #  split the edge, such that it now passes through the inner map.
                    new_map_conn = inner_entry.next_connector(try_name=in_edge.src_conn[4:])
                    new_in_conn = "IN_" + new_map_conn
                    new_out_conn = "OUT_" + new_map_conn
                    new_memlet_inner = dace.Memlet.from_array(
                        in_edge.data.data, sdfg.arrays[in_edge.data.data]
                    )
                    state.add_edge(
                        in_edge.src,
                        in_edge.src_conn,
                        inner_entry,
                        new_in_conn,
                        new_memlet_inner,
                    )
                    state.add_edge(
                        inner_entry,
                        new_out_conn,
                        in_edge.dst,
                        in_edge.dst_conn,
                        copy.deepcopy(in_edge.data),
                    )
                    inner_entry.add_in_connector(new_in_conn)
                    inner_entry.add_out_connector(new_out_conn)
                    state.remove_edge(in_edge)

                else:
                    raise NotImplementedError("Unknown node configuration.")

        # In certain cases it might happen that we need to create an empty
        #  Memlet between the outer map entry and the inner one.
        if state.in_degree(inner_entry) == 0:
            state.add_edge(
                outer_entry,
                None,
                inner_entry,
                None,
                dace.Memlet(),
            )

        # Handle the Map exits
        #  This is simple reconnecting, there would be possibilities for improvements
        #  but we do not use them for now.
        for in_edge in state.in_edges(outer_exit):
            edge_conn = inner_exit.next_connector(in_edge.dst_conn[3:])
            dace_helpers.redirect_edge(
                state=state,
                edge=in_edge,
                new_dst=inner_exit,
                new_dst_conn="IN_" + edge_conn,
            )
            state.add_edge(
                inner_exit,
                "OUT_" + edge_conn,
                outer_exit,
                in_edge.dst_conn,
                copy.deepcopy(in_edge.data),
            )
            inner_exit.add_in_connector("IN_" + edge_conn)
            inner_exit.add_out_connector("OUT_" + edge_conn)

        # There is an invalid cache state in the SDFG, that makes the memlet
        #  propagation fail, to clear the cache we call the hash function.
        #  See: https://github.com/spcl/dace/issues/1703
        _ = sdfg.reset_cfg_list()
        dace_propagation.propagate_memlets_map_scope(sdfg, state, outer_entry)

    def _check_if_blocking_is_favourable(
        self,
        state: dace.SDFGState,
    ) -> bool:
        """Test if the nodes are really independent nodes.

        After the classification the function will examine the set to see if some
        nodes were found that brings no benefit to move out. The classical example
        is a Tasklet that writes a constant into an AccessNode. These kind of
        nodes are filtered out.

        The function returns `True` if it decides that blocking is good and `False`
        otherwise. The function will not modify `self._independent_nodes`.
        """
        assert self._independent_nodes is not None
        assert self._dependent_nodes is None

        # There is nothing to move out so ignore it.
        if len(self._independent_nodes) == 0:
            return False

        # Currently we only filter out Tasklets that do not read any data, which
        #  is the example above, Because of how DaCe works we also subtract all
        #  of its output nodes, that are classified independent.
        # TODO(phimuell): Think if we should expand on that.
        nb_independent_nodes = len(self._independent_nodes)

        for node in self._independent_nodes:
            if isinstance(node, dace_nodes.Tasklet):
                if not all(iedge.data.is_empty() for iedge in state.in_edges(node)):
                    continue
                nb_independent_nodes -= 1
                for oedge in state.out_edges(node):
                    assert isinstance(oedge.dst, dace_nodes.AccessNode)
                    assert oedge.dst in self._independent_nodes
                    nb_independent_nodes -= 1
            assert nb_independent_nodes >= 0

        return nb_independent_nodes > 0
