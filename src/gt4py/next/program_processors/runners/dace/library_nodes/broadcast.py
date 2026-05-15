# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any, Final, Iterable, Sequence

import dace
from dace import (
    data as dace_data,
    library as dace_library,
    nodes as dace_nodes,
    properties as dace_properties,
    subsets as dace_subset,
)
from dace.sdfg import graph as dace_graph
from dace.transformation import transformation as dace_transform

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import lowering as gtx_dace_lowering


_INPUT_NAME: Final[str] = "_inp"
_OUTPUT_NAME: Final[str] = "_outp"


# TODO(edopao): Make sure Philip did the renaming.
@dace_library.node
class Broadcast(dace_nodes.LibraryNode):
    """Implements broadcasting, i.e. replication of data.

    The broadcasting is controlled by `brodcast_in_dims`, which is a list of integer.
    Its length equals the dimensionality of the input date, i.e. there is an entry
    for every dimension of the input. The array describes where a dimension of the
    input ends up in the output. See [here](https://openxla.org/stablehlo/spec#broadcast_in_dim)
    for more information.

    The expansion of a broadcast node will generate a Map. The names of the parameters
    can either specified, through the `params` argument or by passing `None` in which
    case automatically generated names are used.
    It is strongly recommended to specify names for the parameter, because of the
    expansion such that they can be brought into the right order.

    Args:
        broadcast_in_dim: Description of how the dimensions of the input are
            mapped to the dimensions of the output, see class description.
        params: Names that should be used for the expansion. Can either be strings
            or GT4Py `Dimension`s (recommended). If `None` then automatic names
            are used.
    """

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    brodcast_in_dims = dace_properties.ListProperty(element_type=int)
    params = dace_properties.ListProperty(element_type=str, allow_none=True)

    def __init__(
        self,
        name: str,
        broadcast_in_dims: Sequence[int],
        params: Iterable[gtx_common.Dimension | str] | None,
        debuginfo: dace.dtypes.DebugInfo | None = None,
    ):
        super().__init__(name, inputs={_INPUT_NAME}, outputs={_OUTPUT_NAME})

        self.brodcast_in_dims = list(broadcast_in_dims)
        self.debuginfo = debuginfo

        if params is None:
            self.params = None
        else:
            self.params = [
                gtx_dace_lowering.get_map_variable(param)
                if isinstance(param, gtx_common.Dimension)
                else param
                for param in params
            ]

    @property
    def free_symbols(self) -> set[str]:
        # This function is needed to ensure that the names in `params` are not used
        #  as free symbols.
        return set()

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if len(self.brodcast_in_dims) != len(set(self.brodcast_in_dims)):
            raise ValueError("`Broadcast dimensions must be unique")

        # They are two different properties, that are upon initialization the same
        #  but there is no check to ensure that this stays this way.
        if self.name != self.label:
            raise ValueError(
                "Label of `Broadcast` node, `{self.label}`, does not match its name, `{self.name}`."
            )

        # TODO(phimuell): Handle empty Memlets in the input.
        if state.in_degree(self) != 1 or next(iter(state.in_edges(self))).dst_conn != _INPUT_NAME:
            raise ValueError("GT4Py Broadcast node needs exactly one input.")
        if (
            state.out_degree(self) != 1
            or next(iter(state.out_edges(self))).src_conn != _OUTPUT_NAME
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

        if not isinstance(bcast_result_desc, dace_data.Array):
            # In fact it would also be possible to broadcast into a Scalar, but this
            #  does not make much sense.
            raise ValueError(
                f"Can only broadcast into an array, but target was `{type(bcast_result_desc).__name__}`."
            )

        if (self.params is not None) and (len(self.params) != len(bcast_result_desc.shape)):
            raise ValueError(
                f"The number of dimensions of the result shape {len(bcast_result_desc.shape)}"
                f"does not match the nmuber of broadcast parameters {len(self.params)}."
            )

        if isinstance(bcast_value_desc, dace_data.Scalar):
            if len(self.brodcast_in_dims) != 0:
                raise ValueError("For a scalar `broadcast_in_dims` must be empty.")
        else:
            if len(self.brodcast_in_dims) != len(bcast_value_desc.shape):
                raise ValueError(
                    f"`broadcast_in_dims` has {len(self.brodcast_in_dims)} entries,"
                    f" but the value to broadcast had {len(bcast_value_desc.shape)} dimensions."
                )
            if len(bcast_result_desc.shape) < len(bcast_value_desc.shape):
                raise ValueError(
                    f"The value to broadcast has more dimensions ({len(bcast_value_desc.shape)})"
                    f" than the result ({len(bcast_result_desc.shape)})."
                )

            # TODO(phimuell): Improve these tests.
            for bcast_dst_dim in self.brodcast_in_dims:
                if bcast_dst_dim < 0:
                    raise ValueError(f"Negative dimension index ({bcast_dst_dim}) is invalid.")
                if bcast_dst_dim >= len(bcast_result_desc.shape):
                    raise ValueError("Out of range broadcast dim found.")


def inplace_broadcast_expander(
    bcast_node: Broadcast,
    state: dace.SDFGState,
    sdfg: dace.SDFG,
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

    # TODO(edopao): Remember Philip to forbid it, i.e. force specification of it.
    map_params: list[str] = (
        [f"__bcast{dst_dim}" for dst_dim in range(len(output_edge.data.subset))]
        if bcast_node.params is None
        else list(bcast_node.params)
    )

    # NOTE: Creating it that way allows to have numbers as Map bounds for subsets such
    #   as `[a:(a + 5)]`.
    bcast_size = output_edge.data.subset.size()
    output_offset = output_edge.data.subset.min_element()
    output_subset: list[str] = [
        f"({offset}) + ({param})" for offset, param in zip(output_offset, map_params)
    ]

    map_ranges: dict[str, dace_subset.Range] = {
        map_param: dace_subset.Range.from_string(f"0:({dim_size})")
        for map_param, dim_size in zip(map_params, bcast_size, strict=True)
    }

    input_subset: list[str]
    if len(bcast_node.brodcast_in_dims) == 0:
        input_subset = ["0"]
    else:
        bcast_value_offset = input_edge.data.subset.min_element()
        input_subset = [
            f"{map_params[dst_dim]} + ({offset})"
            for dst_dim, offset in zip(bcast_node.brodcast_in_dims, bcast_value_offset, strict=True)
        ]

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

    # NOTE: The order is important. First we need to create the edge outside the Map
    #   and then the one inside. This is important to ensure that the Memlet is
    #   properly initialized, i.e. `_is_data_src` is set.
    state.add_edge(
        mx,
        f"OUT_{output_edge.data.data}",
        output_edge.dst,
        output_edge.dst_conn,
        dace.Memlet(data=output_edge.data.data, subset=copy.deepcopy(output_edge.data.subset)),
    )
    new_output_edge = state.add_edge(
        bcast_tlet,
        "__out",
        mx,
        f"IN_{output_edge.data.data}",
        dace.Memlet(data=output_edge.data.data, subset=", ".join(output_subset)),
    )
    mx.add_scope_connectors(output_edge.data.data)
    assert new_output_edge.data._is_data_src is not None

    # Now delete the node.
    state.remove_node(bcast_node)


@dace_library.register_expansion(Broadcast, "pure")
class BroadcastExpandInlined(dace_transform.ExpandTransformation):
    """Implements pure expansion of the Broadcast library node."""

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(node: Broadcast, state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:

        # NOTE: We have to cheat a here a bit. Actually only parts of the output
        #   would be mapped into the nested SDFG. However, we do not want that.
        #   Instead we will modify the edges on the outside, such that everything
        #   is mapped into the nested SDFG.
        assert isinstance(node, Broadcast)
        assert state.out_degree(node) == 1 and state.in_degree(node) == 1

        nsdfg = dace.SDFG(f"__gt4py_broadcast_expansion_{node.label}")
        bcast_st = nsdfg.add_state(f"__gt4py_broadcast_expansion_{node.label}_state")

        input_edge = next(state.in_edges_by_connector(node, _INPUT_NAME))
        output_edge = next(state.out_edges_by_connector(node, _OUTPUT_NAME))
        bcast_value = input_edge.src
        bcast_result = output_edge.dst

        # Creating the input and output data inside the nested SDFG, such that we can
        #  map them _fully_ (see later) into the nested SDFG.
        bcast_value_inner_data = nsdfg.add_datadesc(_INPUT_NAME, bcast_value.desc(sdfg).clone())
        bcast_result_inner_data = nsdfg.add_datadesc(_OUTPUT_NAME, bcast_result.desc(sdfg).clone())
        nsdfg.arrays[bcast_value_inner_data].transient = False
        nsdfg.arrays[bcast_result_inner_data].transient = False

        inner_bcast_node = copy.deepcopy(node)
        inner_bcast_value_edge = bcast_st.add_edge(
            bcast_st.add_access(bcast_value_inner_data),
            input_edge.src_conn,
            inner_bcast_node,
            "_inp",
            copy.deepcopy(input_edge.data),
        )
        inner_bcast_value_edge.data.data = bcast_value_inner_data

        inner_bcast_result_edge = bcast_st.add_edge(
            inner_bcast_node,
            "_outp",
            bcast_st.add_access(bcast_result_inner_data),
            output_edge.dst_conn,
            copy.deepcopy(output_edge.data),
        )
        inner_bcast_result_edge.data.data = bcast_result_inner_data

        # Now we run the inplace expansion on the node inside the nested SDFG.
        inplace_broadcast_expander(inner_bcast_node, bcast_st, nsdfg)

        # To ensure that the full data is passed into the nested SDFG, which we
        #  assumed because of how we want that everything is mapped into.
        input_edge.data.subset = dace_subset.Range.from_array(bcast_value.desc(sdfg))
        output_edge.data.subset = dace_subset.Range.from_array(bcast_result.desc(sdfg))

        # NOTE: We will not update `nsdfg.symbols`, instead we will rely on
        #   `add_nested_sdfg()` that is called implicitly by the expansion driver.

        return nsdfg
