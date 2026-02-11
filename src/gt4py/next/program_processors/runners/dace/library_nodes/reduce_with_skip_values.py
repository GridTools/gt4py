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
from dace.libraries import standard as dace_stdlib
from dace.sdfg import graph as dace_graph
from dace.transformation import transformation as dace_transform

from gt4py.next import common as gtx_common


_INPUT_NAME: Final[str] = "_in"
_OUTPUT_NAME: Final[str] = "_out"
_MASK_NAME: Final[str] = "_mask"


class ReduceWithSkipValues(dace_stdlib.Reduce):
    """Implements reduction with skip values."""

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    init = dace_properties.SymbolicProperty(default=0)

    def __init__(
        self,
        name: str,
        wcr: str,
        identity: dace.symbolic.SymbolicType,
        init: dace.symbolic.SymbolicType,
        debuginfo: dace.dtypes.DebugInfo | None = None,
    ) -> None:
        super().__init__(
            name,
            wcr,
            None,
            identity,
            dace.dtypes.ScheduleType.Default,
            debuginfo,
            inputs={_INPUT_NAME, _MASK_NAME},
            outputs={_OUTPUT_NAME},
        )
        self.init = init

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        return


@dace_library.register_expansion(ReduceWithSkipValues, "pure")
class ReduceWithSkipValuesExpandInlined(dace_transform.ExpandTransformation):
    """Implements pure expansion of the ReduceWithSkipValues library node."""

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(node: ReduceWithSkipValues, state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        assert len(list(state.in_edges_by_connector(node, _INPUT_NAME))) == 1
        inedge: dace_graph.MultiConnectorEdge = next(state.in_edges_by_connector(node, _INPUT_NAME))
        assert len(list(state.out_edges_by_connector(node, _OUTPUT_NAME))) == 1
        outedge: dace_graph.MultiConnectorEdge = next(
            state.out_edges_by_connector(node, _OUTPUT_NAME)
        )
        assert len(list(state.in_edges_by_connector(node, _MASK_NAME))) == 1
        maskedge: dace_graph.MultiConnectorEdge = next(
            state.in_edges_by_connector(node, _MASK_NAME)
        )
        input_desc = sdfg.arrays[inedge.data.data]
        output_desc = sdfg.arrays[outedge.data.data]
        mask_desc = sdfg.arrays[maskedge.data.data]
        assert len(mask_desc.shape) == 2

        max_neighbors = mask_desc.shape[1]
        assert isinstance(max_neighbors, int) or str(max_neighbors).isdigit()
        assert inedge.data.num_elements() == max_neighbors
        assert maskedge.data.num_elements() == max_neighbors
        assert outedge.data.num_elements() == 1

        local_dim_index = inedge.data.src_subset.size().index(max_neighbors)

        nsdfg = dace.SDFG(node.label)
        nsdfg.add_array(
            _INPUT_NAME,
            (max_neighbors,),
            input_desc.dtype,
            strides=(input_desc.strides[local_dim_index],),
        )
        nsdfg.add_array(
            _MASK_NAME,
            (max_neighbors,),
            mask_desc.dtype,
            strides=(mask_desc.strides[1],),
        )
        nsdfg.add_scalar(_OUTPUT_NAME, output_desc.dtype)
        st_init = nsdfg.add_state("init")
        init_tasklet = st_init.add_tasklet(
            name="write",
            inputs={},
            outputs={"val"},
            code=f"val = {input_desc.dtype}({node.init})",
        )
        st_init.add_edge(
            init_tasklet,
            "val",
            st_init.add_access(_OUTPUT_NAME),
            None,
            dace.Memlet(data=_OUTPUT_NAME, subset="0"),
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
                "val": dace.Memlet(data=_INPUT_NAME, subset="i"),
                "neighbor_idx": dace.Memlet(data=_MASK_NAME, subset="i"),
            },
            code=f"out = val if neighbor_idx != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}",
            outputs={
                "out": dace.Memlet(data=_OUTPUT_NAME, subset="0", wcr=node.wcr, wcr_nonatomic=True),
            },
            external_edges=True,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        return nsdfg
