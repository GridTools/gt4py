# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any, Final, Sequence

import dace
from dace import (
    data as dace_data,
    library as dace_library,
    nodes as dace_nodes,
    properties as dace_properties,
    subsets as dace_sbs,
)
from dace.sdfg import graph as dace_graph
from dace.transformation import transformation as dace_transform


_INPUT_NAME: Final[str] = "_inp"
_OUTPUT_NAME: Final[str] = "_outp"


@dace_library.node
class Broadcast(dace_nodes.LibraryNode):
    """Implements write of a scalar value over an array subset.

    ndims(output) == ndims(value_to_broadcast) + len(broadcast_in_dim)
    Same as XLA.
    broadcast_in_dim[i] describes where dimension `i` of the `value_to_broadcast`
    goes. In case of a scalar it is empty.
    Furthermore the following has to hold:
    ```python
    for i in range(len(broadcast_in_dim):
        assert output.shape[broadcast_in_dim[i]] == value_to_broadcast.shape[i]
    ```

    Args:
        broadcast_in_dim: How to broadcast.

    Todo:
        - While for the output it is probably okay to always require an adjacent
            AccessNode for the input it might be possible to be on the other side
            of a Map.
    """

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    brodcast_in_dims = dace_properties.ListProperty(element_type=int)

    def __init__(
        self,
        name: str,
        broadcast_in_dims: Sequence[int],
        debuginfo: dace.dtypes.DebugInfo | None = None,
    ):
        # TODO(philip, edopao): I would propose to drop `value` then. This makes it
        #   simpler to handle in the transformations.
        super().__init__(name, inputs={_INPUT_NAME}, outputs={_OUTPUT_NAME})

        self.brodcast_in_dims = list(broadcast_in_dims)
        self.debuginfo = debuginfo

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if len(self.brodcast_in_dims) == len(set(self.brodcast_in_dims)):
            raise ValueError("`FCan not broadcast to multiple dimensions at the same time.")

        if state.in_degree(self) != 1 and next(iter(state.in_edges(self))).dst_conn == _INPUT_NAME:
            raise ValueError("GT4Py Broadcast node needs exactly one input.")
        if (
            state.out_degree(self) != 1
            and next(iter(state.out_edges(self))).src_conn == _OUTPUT_NAME
        ):
            raise ValueError("GT4Py Broadcast node needs exactly one output.")

        bcast_value_node: dace_nodes.AccessNode = next(iter(state.in_edges(self))).src
        if not isinstance(bcast_value_node, dace_nodes.AccessNode):
            raise ValueError("Source of broadcasting must be an AccessNode.")
        bcast_value_desc = bcast_value_node.desc(sdfg)
        if isinstance(bcast_value_desc, dace_data.View):
            raise ValueError("Can not broadcast from a view.")

        bcast_result_node: dace_nodes.AccessNode = next(iter(state.out_edges(self))).dst
        if not isinstance(bcast_result_node, dace_nodes.AccessNode):
            raise ValueError("Broadcast result must be an AccessNode.")
        bcast_result_desc = bcast_result_node.desc(sdfg)
        if isinstance(bcast_result_desc, dace_data.View):
            raise ValueError("Broadcast result can not be a view.")

        if isinstance(bcast_value_desc, dace_data.Scalar):
            if len(self.brodcast_in_dims) == 0:
                raise ValueError("For a scalar `broadcast_in_dims` must be empty.")
        else:
            expected_output_dims = len(self.brodcast_in_dims) + len(bcast_value_desc.shape)
            got_output_dims = len(bcast_result_desc.shape)
            if expected_output_dims != got_output_dims:
                raise ValueError(
                    f"Expected output to have {expected_output_dims}, but it only had {got_output_dims}"
                )

            for src_dim, bcast_dst_dim in enumerate(self.brodcast_in_dims):
                if bcast_dst_dim < 0:
                    raise ValueError("Negative broadcast")
                if bcast_dst_dim >= len(bcast_result_desc.shape):
                    raise ValueError("Out of range broadcast dim found.")

                # Only do the size matching test if the sizes are known, as different
                #  symbols can have the same value.
                src_size = bcast_value_desc.shape[src_dim]
                dst_size = bcast_result_desc.shape[bcast_dst_dim]
                if str(src_size).isdigit() and str(dst_size).isdigit() and (src_size != dst_size):
                    raise ValueError("Size mismatch found.")


def inplace_broadcast_expander(
    sdfg: dace.SDFG, state: dace.SDFGState, bcast_node: Broadcast
) -> None:
    """Perform expansion of `bcast_node` inside `state`.

    The main difference between this and the normal expansion transformation is
    that this function does not generate nested SDFG and instead performs the
    expansion in place.
    """

    input_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = next(iter(state.in_edges(bcast_node)))
    output_edge: dace_graph.MultiConnectorEdge[dace.Memlet] = next(
        iter(state.out_edges(bcast_node))
    )

    map_ranges: dict[str, dace_sbs.Range] = {}
    output_subset: list[str] = []
    for dst_dim, sbs in enumerate(output_edge.data.subset):
        assert isinstance(sbs, tuple) and len(sbs) == 3
        output_subset.append(f"__bcast{dst_dim}")
        map_ranges[output_subset[-1]] = dace_sbs.Range(sbs)

    if len(bcast_node.brodcast_in_dims) == 0:
        input_subset = ["0"]
    else:
        # TODO(phimuell): Do we need a correction here because map range can be offsetted.
        #   Because of our requierement it is probably irrelevant, i.e. the dimension is always full.
        input_subset = [f"__bcast{dst_dim}" for dst_dim in bcast_node.brodcast_in_dims]

    me, mx = state.add_map(f"__gt4py_broadcast_map_{bcast_node.name}", ndrange=map_ranges)
    bcast_tlet = state.add_tasklet(
        f"__gt4py_broadcast_tasklet_{bcast_node.name}",
        inputs={"__in"},
        outputs={"__out"},
        code="__out = __in",
    )

    state.add_edge(
        input_edge.src,
        input_edge.src_conn,
        me,
        f"IN_{input_edge.data.data}",
        dace.Memlet.from_array(input_edge.data.data, sdfg.arrays[input_edge.data.data]),
    )
    state.add_edge(
        me,
        f"OUT_{input_edge.data.data}",
        bcast_tlet,
        "__in",
        dace.Memlet(data=input_edge.data.data, subset=", ".join(input_subset)),
    )
    me.add_scope_connectors(input_edge.data.data)

    state.add_edge(
        bcast_tlet,
        "__out",
        mx,
        f"IN_{output_edge.data.data}",
        dace.Memlet(data=output_edge.data.data, subset=", ".join(output_subset)),
    )
    state.add_edge(
        mx,
        f"OUT_{output_edge.data.data}",
        output_edge.dst,
        output_edge.dst_conn,
        dace.Memlet(data=output_edge.data.data, subset=copy.deepcopy(output_edge.data.subset)),
    )
    mx.add_scope_connectors(output_edge.data.data)

    state.remove_node(bcast_node)


@dace_library.register_expansion(Broadcast, "pure")
class BroadcastExpandInlined(dace_transform.ExpandTransformation):
    """Implements pure expansion of the Broadcast library node.

    Todo:
        - In DaCe the expansion must happen in a NestedSDFG. However, this is a bit
            bad, and we actually would need to run simplification again to get rid
            of them and proper process them. There should be a function which
            essentially inlines it inside the SDFG.
    """

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(node: Broadcast, state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        # TODO(phimuell, edopao): I would say a broadcast node should have exactly
        #   one output edge and if we drop `value` exactly one input edge.
        #   This must then also be done in `validate()`.
        assert state.out_degree(node) == 1
        assert isinstance(node, Broadcast)

        nsdfg = dace.SDFG(node.label)
        bcast_st = nsdfg.add_state(f"{node.label}_impl")

        outedge = next(state.out_edges_by_connector(node, _OUTPUT_NAME))
        out_desc = sdfg.arrays[outedge.data.data]
        inner_out_desc = out_desc.clone()
        inner_out_desc.transient = False
        outp = nsdfg.add_datadesc(_OUTPUT_NAME, inner_out_desc)

        dst_subset = outedge.data.get_dst_subset(outedge, state)
        map_params = [f"_i{i}" for i in range(len(dst_subset))]
        out_mem = dace.Memlet(data=outp, subset=",".join(map_params))

        if node.value is None:
            assert len(list(state.in_edges_by_connector(node, _INPUT_NAME))) == 1
            inedge = next(state.in_edges_by_connector(node, _INPUT_NAME))
            inp_desc = sdfg.arrays[inedge.data.data]
            inner_inp_desc = inp_desc.clone()
            inner_inp_desc.transient = False
            inp = nsdfg.add_datadesc(_INPUT_NAME, inner_inp_desc)

            if node.axes:
                index_map = dict(enumerate(map_params))
                inp_subset = ",".join(
                    f"{index_map[i]} + {node.dst_origin[i]} - ({src_origin})"
                    for i, src_origin in zip(node.axes, node.src_origin, strict=True)
                )
            else:
                inp_subset = "0"

            bcast_st.add_mapped_tasklet(
                name=node.label,
                map_ranges=dict(zip(map_params, dst_subset, strict=True)),
                inputs={"__tlet_inp": dace.Memlet(data=inp, subset=inp_subset)},
                code="__tlet_out = __tlet_inp",
                outputs={"__tlet_out": out_mem},
                external_edges=True,
            )
        else:
            bcast_st.add_mapped_tasklet(
                name="broadcast",
                map_ranges=dict(zip(map_params, dst_subset)),
                inputs={},
                code=f"outp = {node.value}",
                outputs={"outp": out_mem},
                external_edges=True,
            )

        return nsdfg
