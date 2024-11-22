# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Any, Container, Optional

import dace
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_common import utility as dace_utils


def gt_make_transients_persistent(
    sdfg: dace.SDFG,
    device: dace.DeviceType,
) -> dict[int, set[str]]:
    """
    Changes the lifetime of certain transients to `Persistent`.

    A persistent lifetime means that the transient is allocated only the very first
    time the SDFG is executed and only deallocated if the underlying `CompiledSDFG`
    object goes out of scope. The main advantage is, that memory must not be
    allocated every time the SDFG is run. The downside is that the SDFG can not be
    called by different threads.

    Args:
        sdfg: The SDFG to process.
        device: The device type.

    Returns:
        A `dict` mapping SDFG IDs to a set of transient arrays that
        were made persistent.

    Note:
        This function is based on a similar function in DaCe. However, the DaCe
        function does, for unknown reasons, also reset the `wcr_nonatomic` property,
        but only for GPU.
    """
    result: dict[int, set[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        fsyms: set[str] = nsdfg.free_symbols
        modify_lifetime: set[str] = set()
        not_modify_lifetime: set[str] = set()

        for state in nsdfg.states():
            for dnode in state.data_nodes():
                if dnode.data in not_modify_lifetime:
                    continue

                if dnode.data in nsdfg.constants_prop:
                    not_modify_lifetime.add(dnode.data)
                    continue

                desc = dnode.desc(nsdfg)
                if not desc.transient or type(desc) not in {dace.data.Array, dace.data.Scalar}:
                    not_modify_lifetime.add(dnode.data)
                    continue
                if desc.storage == dace.StorageType.Register:
                    not_modify_lifetime.add(dnode.data)
                    continue

                if desc.lifetime == dace.AllocationLifetime.External:
                    not_modify_lifetime.add(dnode.data)
                    continue

                try:
                    # The symbols describing the total size must be a subset of the
                    #  free symbols of the SDFG (symbols passed as argument).
                    # NOTE: This ignores the renaming of symbols through the
                    #   `symbol_mapping` property of nested SDFGs.
                    if not set(map(str, desc.total_size.free_symbols)).issubset(fsyms):
                        not_modify_lifetime.add(dnode.data)
                        continue
                except AttributeError:  # total_size is an integer / has no free symbols
                    pass

                # Make it persistent.
                modify_lifetime.add(dnode.data)

        # Now setting the lifetime.
        result[nsdfg.cfg_id] = modify_lifetime - not_modify_lifetime
        for aname in result[nsdfg.cfg_id]:
            nsdfg.arrays[aname].lifetime = dace.AllocationLifetime.Persistent

    return result


def gt_find_constant_arguments(
    call_args: dict[str, Any],
    include: Optional[Container[str]] = None,
) -> dict[str, Any]:
    """Scans the calling arguments for compile time constants.

    The output of this function can be used as input to
    `gt_substitute_compiletime_symbols()`, which then removes these symbols.

    By specifying `include` it is possible to force the function to include
    additional arguments, that would not be matched otherwise. Importantly,
    their value is not checked.

    Args:
        call_args: The full list of arguments that will be passed to the SDFG.
        include: List of arguments that should be included.
    """
    if include is None:
        include = set()
    ret_value: dict[str, Any] = {}

    for name, value in call_args.items():
        if name in include or (dace_utils.is_field_symbol(name) and value == 1):
            ret_value[name] = value

    return ret_value


def is_accessed_downstream(
    start_state: dace.SDFGState,
    sdfg: dace.SDFG,
    data_to_look: str,
    nodes_to_ignore: Optional[set[dace_nodes.AccessNode]] = None,
    states_to_ignore: Optional[set[dace.SDFGState]] = None,
) -> bool:
    """Scans for accesses to the data container `data_to_look`.

    The function will go through states that are reachable from `start_state`
    (included) and test if there is an AccessNode that refers to `data_to_look`.
    It will return `True` the first time it finds such a node.

    The function will ignore all nodes that are listed in `nodes_to_ignore`.
    Furthermore, states listed in `states_to_ignore` will be ignored, i.e.
    handled as they did not exist.

    Args:
        start_state: The state where the scanning starts.
        sdfg: The SDFG on which we operate.
        data_to_look: The data that we want to look for.
        nodes_to_ignore: Ignore these nodes.
        states_to_ignore: Ignore these states.
    """
    seen_states: set[dace.SDFGState] = set()
    to_visit: list[dace.SDFGState] = [start_state]
    ign_dnodes: set[dace_nodes.AccessNode] = nodes_to_ignore or set()
    ign_states: set[dace.SDFGState] = states_to_ignore or set()

    while len(to_visit) > 0:
        state = to_visit.pop()
        seen_states.add(state)
        for dnode in state.data_nodes():
            if dnode.data != data_to_look:
                continue
            if dnode in ign_dnodes:
                continue
            if state.out_degree(dnode) != 0:
                return True  # There is a read operation

        # Look for new states, also scan the interstate edges.
        for out_edge in sdfg.out_edges(state):
            if out_edge.dst in ign_states:
                continue
            if data_to_look in out_edge.data.read_symbols():
                return True
            if out_edge.dst in seen_states:
                continue
            to_visit.append(out_edge.dst)

    return False
