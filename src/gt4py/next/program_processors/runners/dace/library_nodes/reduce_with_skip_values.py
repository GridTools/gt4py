# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any, Final

import dace
from dace import library as dace_library, properties as dace_properties, subsets as dace_subsets
from dace.sdfg import graph as dace_graph
from dace.transformation import transformation as dace_transform

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import sdfg_utils as gtx_dace_utils


@dace.library.node
class ReduceWithSkipValues(dace.sdfg.nodes.LibraryNode):
    """Implements reduction with skip values."""

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    # Properties
    wcr = dace_properties.LambdaProperty(allow_none=True)
    identity = dace_properties.Property(allow_none=True)
    init = dace_properties.Property(allow_none=True)
    input_conn = dace_properties.Property(dtype=str)
    output_conn = dace_properties.Property(dtype=str)
    mask_conn = dace_properties.Property(dtype=str)

    def __init__(
        self,
        name: str,
        wcr: str,
        identity: dace.symbolic.SymbolicType,
        init: dace.symbolic.SymbolicType,
        input_conn: str,
        output_conn: str,
        mask_conn: str,
        debuginfo: dace.dtypes.DebugInfo | None = None,
    ) -> None:
        super().__init__(name, inputs={input_conn, mask_conn}, outputs={output_conn})
        self.wcr = wcr
        self.identity = identity
        self.init = init
        self.input_conn = input_conn
        self.output_conn = output_conn
        self.mask_conn = mask_conn
        self.debuginfo = debuginfo

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if len(list(state.in_edges_by_connector(self, self.input_conn))) != 1:
            raise ValueError(f"Expected exactly one input edge on connector {self.input_conn}.")
        inedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(self, self.input_conn)
        )
        if len(list(state.out_edges_by_connector(self, self.output_conn))) != 1:
            raise ValueError(f"Expected exactly one output edge on connector {self.output_conn}.")
        outedge: dace_graph.MultiConnectorEdge = next(
            state.out_edges_by_connector(self, self.output_conn)
        )
        if len(list(state.in_edges_by_connector(self, self.mask_conn))) != 1:
            raise ValueError(f"Expected exactly one input edge on connector {self.mask_conn}.")
        maskedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(self, self.mask_conn)
        )

        mask_desc = sdfg.arrays[maskedge.data.data]
        if len(mask_desc.shape) != 2:
            raise ValueError(f"Invalid shape {mask_desc.shape} of mask array, expected 2d array.")
        max_neighbors = mask_desc.shape[1]
        if not gtx_dace_utils.is_compile_time_size(max_neighbors):
            raise ValueError(
                f"Invalid shape {mask_desc.shape} of mask array, expected constant neighbors size."
            )
        if (
            inedge.data.num_elements() != max_neighbors
            or inedge.data.src_subset.size().count(max_neighbors) != 1
        ):
            raise ValueError(f"Invalid memlet on input connector {self.input_conn}.")
        if (
            maskedge.data.num_elements() != max_neighbors
            or maskedge.data.src_subset.size().count(max_neighbors) != 1
        ):
            raise ValueError(f"Invalid memlet on input connector {self.mask_conn}.")
        if outedge.data.num_elements() != 1:
            raise ValueError(f"Invalid memlet on output connector {self.output_conn}.")


_LOCAL_INDEX: Final[dace.symbol] = dace.symbol("__reduce_local_idx")
"""Map parameter that iterates over the local dimension in the expansion."""


def _as_global(desc: dace.data.Data) -> dace.data.Data:
    """Returns a non-transient copy of `desc`, to be used as nested SDFG connector."""
    global_desc = copy.deepcopy(desc)
    global_desc.transient = False
    return global_desc


@dace_library.register_expansion(ReduceWithSkipValues, "pure")
class ReduceWithSkipValuesExpandInlined(dace_transform.ExpandTransformation):
    """Implements pure expansion of the ReduceWithSkipValues library node."""

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(node: ReduceWithSkipValues, state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        assert len(list(state.in_edges_by_connector(node, node.input_conn))) == 1
        inedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(node, node.input_conn)
        )
        assert len(list(state.out_edges_by_connector(node, node.output_conn))) == 1
        outedge: dace_graph.MultiConnectorEdge = next(
            state.out_edges_by_connector(node, node.output_conn)
        )
        assert len(list(state.in_edges_by_connector(node, node.mask_conn))) == 1
        maskedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(node, node.mask_conn)
        )
        input_desc = sdfg.arrays[inedge.data.data]
        output_desc = sdfg.arrays[outedge.data.data]
        mask_desc = sdfg.arrays[maskedge.data.data]
        assert len(mask_desc.shape) == 2
        max_neighbors = mask_desc.shape[1]
        assert gtx_dace_utils.is_compile_time_size(max_neighbors)

        # In validation, we already checked that the input subset collects exactly
        #  `max_neighbors` elements along one dimension.
        local_dim_index = inedge.data.src_subset.size().index(max_neighbors)

        # The connectors of the nested SDFG have to be equivalent to the data connected
        #  to them (see `NestedSDFG.validate()`). Thus, they are copies of the outer
        #  data descriptors, and inside they are accessed with the same indices as
        #  outside, where the local dimension is iterated by the map parameter `_LOCAL_INDEX`.
        #  This way, a later change of the strides of the outer data is propagated
        #  to the connectors, without the need of views.
        def make_local_subset(subset: dace_subsets.Range, local_index: int) -> dace_subsets.Range:
            local_subset = copy.deepcopy(subset)
            local_start = local_subset[local_index][0]
            local_subset[local_index] = (local_start + _LOCAL_INDEX, local_start + _LOCAL_INDEX, 1)
            return local_subset

        input_subset = make_local_subset(inedge.data.src_subset, local_dim_index)
        mask_subset = make_local_subset(maskedge.data.src_subset, 1)

        nsdfg = dace.SDFG(node.label)
        inp = node.input_conn
        nsdfg.add_datadesc(inp, _as_global(input_desc))
        mask = node.mask_conn
        nsdfg.add_datadesc(mask, _as_global(mask_desc))
        outp = node.output_conn
        nsdfg.add_datadesc(outp, _as_global(output_desc))
        output_subset = (
            "0" if isinstance(output_desc, dace.data.Scalar) else str(outedge.data.dst_subset)
        )
        st_init = nsdfg.add_state("init")
        init_tasklet = st_init.add_tasklet(
            name="write",
            inputs={},
            outputs={"__tlet_out": None},
            code=f"__tlet_out = {input_desc.dtype}({node.init})",
        )
        st_init.add_edge(
            init_tasklet,
            "__tlet_out",
            st_init.add_access(outp),
            None,
            dace.Memlet(data=outp, subset=output_subset),
        )
        st_reduce = nsdfg.add_state_after(st_init, "compute")
        # Fill skip values in local dimension with the reduce identity value
        skip_value = f"{input_desc.dtype}({node.identity})"
        # Since this map operates on a pure local dimension, we explicitly set sequential
        # schedule and we set the flag 'wcr_nonatomic=True' on the write memlet.
        # TODO(phimuell): decide if auto-optimizer should reset `wcr_nonatomic` properties, as DaCe does.
        st_reduce.add_mapped_tasklet(
            name="reduce_with_skip_values",
            map_ranges={str(_LOCAL_INDEX): f"0:{max_neighbors}"},
            inputs={
                "__tlet_inp": dace.Memlet(data=inp, subset=input_subset),
                "__tlet_mask": dace.Memlet(data=mask, subset=mask_subset),
            },
            code=f"__tlet_out = __tlet_inp if __tlet_mask != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}",
            outputs={
                "__tlet_out": dace.Memlet(
                    data=outp, subset=output_subset, wcr=node.wcr, wcr_nonatomic=True
                ),
            },
            external_edges=True,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        return nsdfg
