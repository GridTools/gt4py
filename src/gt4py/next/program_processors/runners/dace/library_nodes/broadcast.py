# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Final, Sequence

import dace
from dace import library as dace_library, nodes as dace_nodes, properties as dace_properties
from dace.transformation import transformation as dace_transform


_INPUT_NAME: Final[str] = "_inp"
_OUTPUT_NAME: Final[str] = "_outp"


@dace_library.node
class Broadcast(dace_nodes.LibraryNode):
    """Implements write of a scalar value over an array subset."""

    implementations: Final[dict[str, dace_transform.ExpandTransformation]] = {}
    default_implementation: Final[str | None] = "pure"

    axes = dace_properties.ListProperty(element_type=int)
    src_origin = dace_properties.ListProperty(element_type=str)
    dst_origin = dace_properties.ListProperty(element_type=str)
    value = dace_properties.Property(allow_none=True)

    def __init__(
        self,
        name: str,
        axes: Sequence[int] | None = None,
        src_origin: Sequence[dace.symbolic.SymbolicType] | None = None,
        dst_origin: Sequence[dace.symbolic.SymbolicType] | None = None,
        value: dace.symbolic.SymbolicType | None = None,
        debuginfo: dace.dtypes.DebugInfo | None = None,
    ):
        inputs = {_INPUT_NAME} if value is None else None
        super().__init__(name, inputs=inputs, outputs={_OUTPUT_NAME})

        self.axes = [] if axes is None else list(axes)
        self.src_origin = [] if src_origin is None else [str(o) for o in src_origin]
        self.dst_origin = [] if dst_origin is None else [str(o) for o in dst_origin]
        self.value = value
        self.debuginfo = debuginfo

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        if any(i < 0 for i in self.axes):
            raise ValueError("Invalid negative axis value.")

        if len(self.axes) != len(set(self.axes)):
            raise ValueError("Axes must be unique.")

        assert len(list(state.out_edges_by_connector(self, _OUTPUT_NAME))) == 1
        outedge = next(state.out_edges_by_connector(self, _OUTPUT_NAME))
        if not isinstance(outedge.dst, dace_nodes.AccessNode):
            raise ValueError("Output node must be an access node.")

        if self.value is None:  # expect an input connection
            assert len(list(state.in_edges_by_connector(self, _INPUT_NAME))) == 1
            inedge = next(state.in_edges_by_connector(self, _INPUT_NAME))
            if not isinstance(inedge.src, dace_nodes.AccessNode):
                raise ValueError("Input node must be an access node.")

            if len(self.axes) == 0:
                if inedge.src.desc(sdfg).shape != (1,):
                    raise ValueError("Axes cannot be None with array source.")
            else:
                if len(self.axes) != len(inedge.src.desc(sdfg).shape):
                    raise ValueError("The provided axes are incompatible with source shape.")
                elif max(self.axes) >= len(outedge.dst.desc(sdfg).shape):
                    raise ValueError("The provided axes are incompatible with destination shape.")
                if len(self.src_origin) != len(self.axes):
                    raise ValueError("Invalid source origin.")
                elif len(self.dst_origin) != len(outedge.dst.desc(sdfg).shape):
                    raise ValueError("Invalid destination origin.")
        else:  # broadcast of a literal value
            if len(list(state.in_edges_by_connector(self, _INPUT_NAME))) != 0:
                raise ValueError("Unexpected input connection with literal value.")
            if self.axes:
                raise ValueError("Unexpected domain axes with literal value.")


@dace_library.register_expansion(Broadcast, "pure")
class BroadcastExpandInlined(dace_transform.ExpandTransformation):
    """Implements pure expansion of the Broadcast library node."""

    environments: Final[list[Any]] = []

    @staticmethod
    def expansion(
        node: Broadcast, parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.SDFG:
        sdfg = dace.SDFG(node.label)
        state = sdfg.add_state(f"{node.label}_impl")

        assert len(list(parent_state.out_edges_by_connector(node, _OUTPUT_NAME))) == 1
        outedge = next(parent_state.out_edges_by_connector(node, _OUTPUT_NAME))
        out_desc = parent_sdfg.arrays[outedge.data.data]
        inner_out_desc = out_desc.clone()
        inner_out_desc.transient = False
        outp = sdfg.add_datadesc(_OUTPUT_NAME, inner_out_desc)

        dst_subset = outedge.data.get_dst_subset(outedge, parent_state)
        map_params = [f"_i{i}" for i in range(len(dst_subset))]
        out_mem = dace.Memlet(data=outp, subset=",".join(map_params))

        if node.value is None:
            assert len(list(parent_state.in_edges_by_connector(node, _INPUT_NAME))) == 1
            inedge = next(parent_state.in_edges_by_connector(node, _INPUT_NAME))
            inp_desc = parent_sdfg.arrays[inedge.data.data]
            inner_inp_desc = inp_desc.clone()
            inner_inp_desc.transient = False
            inp = sdfg.add_datadesc(_INPUT_NAME, inner_inp_desc)

            if node.axes:
                index_map = dict(enumerate(map_params))
                inp_subset = ",".join(
                    f"{index_map[i]} + {node.dst_origin[i]} - {src_origin}"
                    for i, src_origin in zip(node.axes, node.src_origin, strict=True)
                )
            else:
                inp_subset = "0"

            state.add_mapped_tasklet(
                name=node.label,
                map_ranges=dict(zip(map_params, dst_subset, strict=True)),
                inputs={"inp": dace.Memlet(data=inp, subset=inp_subset)},
                code="outp = inp",
                outputs={"outp": out_mem},
                external_edges=True,
            )
        else:
            state.add_mapped_tasklet(
                name="broadcast",
                map_ranges=dict(zip(map_params, dst_subset)),
                inputs={},
                code=f"outp = {node.value}",
                outputs={"outp": out_mem},
                external_edges=True,
            )

        return sdfg
