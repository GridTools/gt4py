# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, TypeAlias, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes
from dace.transformation import pass_pipeline as dace_ppl


AccessLocation: TypeAlias = tuple[dace.SDFGState, dace_nodes.AccessNode]
"""An AccessNode and the state it is located in.
"""


@dace_properties.make_properties
class MultiStateGlobalSelfCopyElimination(dace_transformation.Pass):
    """Removes self copying across different states.

    This transformation is very similar to `SingleStateGlobalSelfCopyElimination`, but
    addresses a slightly different case. Assume we have the pattern `(G) -> (T)`
    in one state, i.e. the global data `G` is copied into a transient. In another
    state, we have the pattern `(T) -> (G)`, i.e. the data is written back.

    If the following conditions are satisfied, this transformation will remove all
    writes to `G`:
    - The only write access to `G` happens in the `(T) -> (G)` pattern. ADR-18
        guarantees, that if `G` is used as an input and output it must be pointwise.
        Thus there is no weird shifting.

    If the only usage of `T` is to write into `G` then the transient `T` will be
    removed.

    Note that this transformation does not consider the subsets of the writes from
    `T` to `G` because ADR-18 guarantees to us, that _if_ `G` is a genuine input
    and output, then the `G` read and write subsets have the exact same range.
    If `G` is not an output then any mutating changes to `G` would be invalid.

    Todo:
        - Implement the pattern `(G) -> (T) -> (G)` which is handled currently by
            `SingleStateGlobalSelfCopyElimination`, see `_classify_candidate()` and
            `_remove_writes_to_global()` for more.
        - Make it more efficient such that the SDFG is not scanned multiple times.
    """

    def modifies(self) -> dace_ppl.Modifies:
        return dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes

    def should_reapply(self, modified: dace_ppl.Modifies) -> bool:
        return modified & (dace_ppl.Modifies.Memlets | dace_ppl.Modifies.AccessNodes)

    def apply_pass(
        self, sdfg: dace.SDFG, pipeline_results: dict[str, Any]
    ) -> Optional[dict[dace.SDFG, set[str]]]:
        """Applies the pass.

        The function will return a `dict` that contains for every SDFG, the name
        of the processed data descriptors. If a name refers to a global memory,
        then it means that all write backs, i.e. `(T) -> (G)` patterns, have
        been removed for that `G`. If the name refers to a data descriptor that no
        longer exists, then it means that the write `(G) -> (T)` was also eliminated.
        Currently there is no possibility to identify which transient name belonged
        to a global name.
        """
        result: dict[dace.SDFG, set[str]] = dict()
        for nsdfg in sdfg.all_sdfgs_recursive():
            single_level_res: set[str] = self._process_sdfg(nsdfg, pipeline_results)
            if single_level_res:
                result[nsdfg] = single_level_res

        return result if result else None

    def _process_sdfg(
        self,
        sdfg: dace.SDFG,
        pipeline_results: dict[str, Any],
    ) -> set[str]:
        """Processes the SDFG in a not recursive way, it returns the set of transients
        that have been eliminated.
        """
        redundant_transients = self._find_redundant_transients(sdfg)
        if len(redundant_transients) == 0:
            return set()

        for global_data, write_locations, read_locations in redundant_transients.values():
            self._remove_transient_at_definition_point(global_data, write_locations)
            self._remove_transient_at_using_location(
                sdfg,
                global_data,
                read_locations,
            )

        return set(redundant_transients.keys())

    def _remove_transient_at_using_location(
        self,
        sdfg: dace.SDFG,
        global_data: str,
        read_locations: list[AccessLocation],
    ) -> None:
        """Removes the write back from the transient into the global.

        If the global node has become isolated it will be removed. In addition the
        function will also remove the transient data from the SDFG.
        """
        for state, transient_ac in read_locations:
            assert state.in_degree(transient_ac) == 0
            assert state.out_degree(transient_ac) == 1

            transient_neighbours = [oedge.dst for oedge in state.out_edges(transient_ac)]
            state.remove_node(transient_ac)

            for transient_neighbour in transient_neighbours:
                if state.degree(transient_neighbour) == 0:
                    state.remove_node(transient_neighbour)

            if transient_ac.data in sdfg.arrays:
                sdfg.remove_data(transient_ac.data)

    def _remove_transient_at_definition_point(
        self,
        global_data: str,
        write_locations: list[AccessLocation],
    ) -> None:
        """Clean up the transient at its definition point.

        The function removes the write from the global into the transient. The
        function will also remove any node that has become isolated.
        However, the function will not remove the transient data from the registry.
        """
        for state, transient_ac in write_locations:
            assert state.in_degree(transient_ac) == 1
            assert state.out_degree(transient_ac) == 0

            transient_neighbours = [iedge.src for iedge in state.in_edges(transient_ac)]
            state.remove_node(transient_ac)

            for transient_neighbour in transient_neighbours:
                if state.degree(transient_neighbour) == 0:
                    state.remove_node(transient_neighbour)

    def _find_redundant_transients(
        self,
        sdfg: dace.SDFG,
    ) -> dict[str, tuple[str, list[AccessLocation], list[AccessLocation]]]:
        """Find all redundant transients that can be eliminated.

        The function returns a `dict` mapping the name of the transient data to a tuple
        of length 3. The first element is the name of the global data that defines
        the transient. The second element are the locations where the transient is
        defined, i.e. written to and in the third element are the locations where the
        transient is used.
        """

        # Scan all transients and find their location.
        possible_redundant_transients: dict[
            str, tuple[list[AccessLocation], list[AccessLocation]]
        ] = {}
        for data_name, desc in sdfg.arrays.items():
            if not desc.transient:
                continue
            write_read_locations = self._find_exclusive_read_and_write_locations_of(sdfg, data_name)
            if write_read_locations is not None and (len(write_read_locations[1]) != 0):
                possible_redundant_transients[data_name] = write_read_locations

        # Nothing was found.
        if len(possible_redundant_transients) == 0:
            return {}

        # Determine the associated global data.
        redundant_transients_and_associated_data: dict[
            str, tuple[str, list[AccessLocation], list[AccessLocation]]
        ] = {}
        involved_global_data: set[str] = set()
        for transient_data, (
            write_locations,
            read_locations,
        ) in possible_redundant_transients.items():
            associated_global_data = self._filter_candidate(
                sdfg,
                transient_data,
                write_locations,
                read_locations,
            )
            if associated_global_data is not None:
                # This is for simplifying implementation.
                assert associated_global_data not in involved_global_data
                involved_global_data.add(associated_global_data)
                redundant_transients_and_associated_data[transient_data] = (
                    associated_global_data,
                    write_locations,
                    read_locations,
                )

        return redundant_transients_and_associated_data

    def _filter_candidate(
        self,
        sdfg: dace.SDFG,
        transient_data: str,
        write_locations: list[AccessLocation],
        read_locations: list[AccessLocation],
    ) -> Union[None, str]:
        """Test if the transient can be eliminated.

        The function tests if transient data can be eliminated and be replaced by a
        global data. If this is the case the function returns the name of the data
        and if this is not possible `None`.
        """
        global_data: Union[None, str] = None

        # We now must look at what defines the transient. For that we look at what
        #  defines it. For that there must be global data that writes into it.
        # TODO(phimuell): To better handle `concat_where` also allows that more
        #   producers are allowed, also differently.
        # TODO(phimuell): In `concat_where` we are using `dynamic` Memlets, they should
        #   also be checked.
        for state, transient_access_node in write_locations:
            for iedge in state.in_edges(transient_access_node):
                src_node = iedge.src
                if not isinstance(src_node, dace_nodes.AccessNode):
                    return None

                if src_node.desc(sdfg).transient:
                    # Non global data writes into the transient.
                    # TODO(phimuell): Lift this.
                    return None

                # TODO(phimuell): Extend this such that there could be multiple
                #   connection from the same global.
                if global_data is None:
                    global_data = src_node.data
                else:
                    return None

        # No global data defines the transient.
        if global_data is None:
            return None

        # We now check that the transient is used to write back to the global.
        #  Currently we only allow that there is a single connection.
        #  The main issue with allowing multiple connections is that we can no
        #  longer be sure that everything is written. I encountered some dead
        #  writes once.
        assert global_data is not None
        for state, transient_access_node in read_locations:
            assert state.in_degree(transient_access_node) == 0
            if state.out_degree(transient_access_node) != 1:
                return None
            if not all(
                isinstance(oedge.dst, dace_nodes.AccessNode) and oedge.dst.data == global_data
                for oedge in state.out_edges(transient_access_node)
            ):
                return None

        return global_data

    def _find_exclusive_read_and_write_locations_of(
        self,
        sdfg: dace.SDFG,
        data_name: str,
    ) -> Union[None, tuple[list[AccessLocation], list[AccessLocation]]]:
        """The function finds all locations were `data_name` is written and read.

        The function will scan the SDFG and returns all places where `data_name` is
        written and where it is read from. If there is however, a location where the
        data is read and written to in the same place then the function returns
        `None`.

        In essence this function returns the set of all possible matches the
        transformation is looking for, but further processing has to be performed.
        """
        read_locations: list[AccessLocation] = []
        write_locations: list[AccessLocation] = []

        for state in sdfg.states():
            for dnode in state.data_nodes():
                if dnode.data != data_name:
                    continue
                out_deg = state.out_degree(dnode)
                in_deg = state.in_degree(dnode)

                # This is not the pattern we are looking for.
                if out_deg > 0 and in_deg > 0:
                    return None
                elif out_deg > 0:
                    read_locations.append((state, dnode))
                else:
                    assert in_deg > 0
                    write_locations.append((state, dnode))

        return (write_locations, read_locations)
