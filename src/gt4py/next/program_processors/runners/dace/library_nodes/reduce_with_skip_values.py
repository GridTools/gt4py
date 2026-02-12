# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Final

import dace
from dace import library as dace_library, properties as dace_properties
from dace.sdfg import graph as dace_graph
from dace.transformation import transformation as dace_transform

from gt4py.next import common as gtx_common


@dace.library.node
class ReduceWithSkipValues(dace.sdfg.nodes.LibraryNode):
    """Implements reduction with skip values."""

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    # Properties
    wcr = dace_properties.LambdaProperty(default="lambda a, b: a")
    identity = dace_properties.SymbolicProperty(default=0, to_json=lambda x: str(x))
    init = dace_properties.SymbolicProperty(default=0, to_json=lambda x: str(x))
    input_conn = dace_properties.Property(default="_in")
    output_conn = dace_properties.Property(default="_out")
    mask_conn = dace_properties.Property(default="_mask")

    def __init__(
        self,
        name: str,
        wcr: str,
        identity: dace.symbolic.SymbolicType,
        init: dace.symbolic.SymbolicType,
        debuginfo: dace.dtypes.DebugInfo | None = None,
        input_conn: str | None = None,
        output_conn: str | None = None,
        mask_conn: str | None = None,
    ) -> None:
        if input_conn is None:
            input_conn = self.input_conn
        else:
            self.input_conn = input_conn

        if output_conn is None:
            output_conn = self.output_conn
        else:
            self.output_conn = output_conn

        if mask_conn is None:
            mask_conn = self.mask_conn
        else:
            self.mask_conn = mask_conn

        super().__init__(name, inputs={input_conn, mask_conn}, outputs={output_conn})
        self.wcr = wcr
        self.identity = identity
        self.init = init
        self.debuginfo = debuginfo

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        assert len(list(state.in_edges_by_connector(self, self.input_conn))) == 1
        inedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(self, self.input_conn)
        )
        assert len(list(state.out_edges_by_connector(self, self.output_conn))) == 1
        outedge: dace_graph.MultiConnectorEdge = next(
            state.out_edges_by_connector(self, self.output_conn)
        )
        assert len(list(state.in_edges_by_connector(self, self.mask_conn))) == 1
        maskedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(self, self.mask_conn)
        )

        mask_desc = sdfg.arrays[maskedge.data.data]
        if len(mask_desc.shape) != 2:
            raise ValueError(f"Invalid shape {mask_desc.shape} of mask array, expected 2d array.")
        max_neighbors = mask_desc.shape[1]
        if not (isinstance(max_neighbors, int) or str(max_neighbors).isdigit()):
            raise ValueError(
                f"Invalid shape {mask_desc.shape} of mask array, expected constant neighbors size."
            )
        if inedge.data.num_elements() != max_neighbors:
            raise ValueError(f"Invalid memlet on input connector {self.input_conn}.")
        if maskedge.data.num_elements() != max_neighbors:
            raise ValueError(f"Invalid memlet on input connector {self.mask_conn}.")
        if outedge.data.num_elements() != 1:
            raise ValueError(f"Invalid memlet on output connector {self.output_conn}.")


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
        assert isinstance(max_neighbors, int) or str(max_neighbors).isdigit()

        local_dim_index = inedge.data.src_subset.size().index(max_neighbors)

        nsdfg = dace.SDFG(node.label)
        inp, _ = nsdfg.add_array(
            node.input_conn,
            (max_neighbors,),
            input_desc.dtype,
            strides=(input_desc.strides[local_dim_index],),
        )
        mask, _ = nsdfg.add_array(
            node.mask_conn,
            (max_neighbors,),
            mask_desc.dtype,
            strides=(mask_desc.strides[1],),
        )
        outp, _ = nsdfg.add_scalar(node.output_conn, output_desc.dtype)
        st_init = nsdfg.add_state("init")
        init_tasklet = st_init.add_tasklet(
            name="write",
            inputs={},
            outputs={"__tlet_out"},
            code=f"__tlet_out = {input_desc.dtype}({node.init})",
        )
        st_init.add_edge(
            init_tasklet,
            "__tlet_out",
            st_init.add_access(outp),
            None,
            dace.Memlet(data=outp, subset="0"),
        )
        st_reduce = nsdfg.add_state_after(st_init, "compute")
        # Fill skip values in local dimension with the reduce identity value
        skip_value = f"{input_desc.dtype}({node.identity})"
        # Since this map operates on a pure local dimension, we explicitly set sequential
        # schedule and we set the flag 'wcr_nonatomic=True' on the write memlet.
        # TODO(phimuell): decide if auto-optimizer should reset `wcr_nonatomic` properties, as DaCe does.
        st_reduce.add_mapped_tasklet(
            name="reduce_with_skip_values",
            map_ranges={"i": f"0:{max_neighbors}"},
            inputs={
                "__tlet_inp": dace.Memlet(data=inp, subset="i"),
                "__tlet_mask": dace.Memlet(data=mask, subset="i"),
            },
            code=f"__tlet_out = __tlet_inp if __tlet_mask != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}",
            outputs={
                "__tlet_out": dace.Memlet(data=outp, subset="0", wcr=node.wcr, wcr_nonatomic=True),
            },
            external_edges=True,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        return nsdfg
