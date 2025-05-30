# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import dace

from gt4py import eve
from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.dace import daceir as dcir, prefix
from gt4py.cartesian.gtc.dace.nodes import StencilComputation
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gt4py.cartesian.gtc.dace.utils import (
    compute_dcir_access_infos,
    get_dace_debuginfo,
    make_dace_subset,
)
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import (
    AccessCollector,
    compute_horizontal_block_extents,
)


class OirSDFGBuilder(eve.NodeVisitor):
    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        current_state: dace.SDFGState
        decls: Dict[str, oir.Decl]
        block_extents: Dict[int, Extent]
        access_infos: Dict[str, dcir.FieldAccessInfo]
        loop_counter: int = 0

        def __init__(self, stencil: oir.Stencil):
            self.sdfg = dace.SDFG(stencil.name)
            self.current_state = self.sdfg.add_state(is_start_block=True)
            self.decls = {decl.name: decl for decl in stencil.params + stencil.declarations}
            self.block_extents = compute_horizontal_block_extents(stencil)

            self.access_infos = compute_dcir_access_infos(
                stencil,
                oir_decls=self.decls,
                block_extents=lambda he: self.block_extents[id(he)],
                collect_read=True,
                collect_write=True,
                include_full_domain=True,
            )

        def make_shape(self, field):
            if field not in self.access_infos:
                return [
                    axis.domain_dace_symbol()
                    for axis in dcir.Axis.dims_3d()
                    if self.decls[field].dimensions[axis.to_idx()]
                ] + [d for d in self.decls[field].data_dims]
            return self.access_infos[field].shape + self.decls[field].data_dims

        def make_input_dace_subset(self, node, field):
            local_access_info = compute_dcir_access_infos(
                node,
                collect_read=True,
                collect_write=False,
                block_extents=lambda he: self.block_extents[id(he)],
                oir_decls=self.decls,
            )[field]
            for axis in local_access_info.variable_offset_axes:
                local_access_info = local_access_info.clamp_full_axis(axis)

            return self._make_dace_subset(local_access_info, field)

        def make_output_dace_subset(self, node, field):
            local_access_info = compute_dcir_access_infos(
                node,
                collect_read=False,
                collect_write=True,
                block_extents=lambda he: self.block_extents[id(he)],
                oir_decls=self.decls,
            )[field]
            for axis in local_access_info.variable_offset_axes:
                local_access_info = local_access_info.clamp_full_axis(axis)

            return self._make_dace_subset(local_access_info, field)

        def _make_dace_subset(self, local_access_info, field):
            global_access_info = self.access_infos[field]
            return make_dace_subset(
                global_access_info, local_access_info, self.decls[field].data_dims
            )

    def _vloop_name(self, node: oir.VerticalLoop, ctx: OirSDFGBuilder.SDFGContext) -> str:
        sdfg_name = ctx.sdfg.name
        counter = ctx.loop_counter
        ctx.loop_counter += 1

        return f"{sdfg_name}_vloop_{counter}_{node.loop_order}_{id(node)}"

    def visit_VerticalLoop(self, node: oir.VerticalLoop, *, ctx: OirSDFGBuilder.SDFGContext):
        declarations = {
            acc.name: ctx.decls[acc.name]
            for acc in node.walk_values().if_isinstance(oir.FieldAccess, oir.ScalarAccess)
            if acc.name in ctx.decls
        }
        library_node = StencilComputation(
            name=self._vloop_name(node, ctx),
            extents=ctx.block_extents,
            declarations=declarations,
            oir_node=node,
        )

        state = ctx.sdfg.add_state_after(ctx.current_state)
        ctx.current_state = state
        state.add_node(library_node)

        access_collection = AccessCollector.apply(node)

        for field in access_collection.read_fields():
            access_node = state.add_access(field, debuginfo=get_dace_debuginfo(declarations[field]))
            connector_name = f"{prefix.CONNECTOR_IN}{field}"
            library_node.add_in_connector(connector_name)
            subset = ctx.make_input_dace_subset(node, field)
            state.add_edge(
                access_node, None, library_node, connector_name, dace.Memlet(field, subset=subset)
            )

        for field in access_collection.write_fields():
            access_node = state.add_access(field, debuginfo=get_dace_debuginfo(declarations[field]))
            connector_name = f"{prefix.CONNECTOR_OUT}{field}"
            library_node.add_out_connector(connector_name)
            subset = ctx.make_output_dace_subset(node, field)
            state.add_edge(
                library_node, connector_name, access_node, None, dace.Memlet(field, subset=subset)
            )

    def visit_Stencil(self, node: oir.Stencil):
        ctx = OirSDFGBuilder.SDFGContext(stencil=node)
        for param in node.params:
            if isinstance(param, oir.FieldDecl):
                dim_strs = [d for i, d in enumerate("IJK") if param.dimensions[i]] + [
                    f"d{d}" for d in range(len(param.data_dims))
                ]
                ctx.sdfg.add_array(
                    param.name,
                    shape=ctx.make_shape(param.name),
                    strides=[
                        dace.symbolic.pystr_to_symbolic(f"__{param.name}_{dim}_stride")
                        for dim in dim_strs
                    ],
                    dtype=data_type_to_dace_typeclass(param.dtype),
                    transient=False,
                    debuginfo=get_dace_debuginfo(param),
                )
            else:
                ctx.sdfg.add_symbol(param.name, stype=data_type_to_dace_typeclass(param.dtype))

        for decl in node.declarations:
            dim_strs = [d for i, d in enumerate("IJK") if decl.dimensions[i]] + [
                f"d{d}" for d in range(len(decl.data_dims))
            ]
            ctx.sdfg.add_array(
                decl.name,
                shape=ctx.make_shape(decl.name),
                strides=[
                    dace.symbolic.pystr_to_symbolic(f"__{decl.name}_{dim}_stride")
                    for dim in dim_strs
                ],
                dtype=data_type_to_dace_typeclass(decl.dtype),
                transient=True,
                lifetime=dace.AllocationLifetime.Persistent,
                debuginfo=get_dace_debuginfo(decl),
            )
        self.visit(node.vertical_loops, ctx=ctx)
        ctx.sdfg.validate()
        return ctx.sdfg
