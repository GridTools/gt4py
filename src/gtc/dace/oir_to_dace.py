# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
from typing import Dict

import dace
import dace.properties
import dace.subsets

import eve
import gtc.oir as oir
from gtc import daceir as dcir
from gtc.dace.nodes import StencilComputation
from gtc.dace.symbol_utils import data_type_to_dace_typeclass
from gtc.dace.utils import compute_dcir_access_infos, make_dace_subset
from gtc.definitions import Extent
from gtc.passes.oir_optimizations.utils import AccessCollector, compute_horizontal_block_extents


class OirSDFGBuilder(eve.NodeVisitor):
    @dataclass
    class SDFGContext:
        sdfg: dace.SDFG
        last_state: dace.SDFGState
        decls: Dict[str, oir.Decl]
        block_extents: Dict[int, Extent]
        access_infos: Dict[str, dcir.FieldAccessInfo]

        def __init__(self, stencil: oir.Stencil):
            self.sdfg = dace.SDFG(stencil.name)
            self.last_state = self.sdfg.add_state(is_start_state=True)
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

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, ctx: "OirSDFGBuilder.SDFGContext", **kwargs
    ):
        declarations = {
            acc.name: ctx.decls[acc.name]
            for acc in node.walk_values().if_isinstance(oir.FieldAccess, oir.ScalarAccess)
            if acc.name in ctx.decls
        }
        library_node = StencilComputation(
            name=f"{ctx.sdfg.name}_computation_{id(node)}",
            extents=ctx.block_extents,
            declarations=declarations,
            oir_node=node,
        )

        state = ctx.sdfg.add_state()
        ctx.sdfg.add_edge(ctx.last_state, state, dace.InterstateEdge())
        ctx.last_state = state

        state.add_node(library_node)

        access_collection = AccessCollector.apply(node)

        for field in access_collection.read_fields():
            access_node = state.add_access(field, debuginfo=dace.DebugInfo(0))
            library_node.add_in_connector("__in_" + field)
            subset = ctx.make_input_dace_subset(node, field)
            state.add_edge(
                access_node,
                None,
                library_node,
                "__in_" + field,
                dace.Memlet(field, subset=subset),
            )
        for field in access_collection.write_fields():
            access_node = state.add_access(field, debuginfo=dace.DebugInfo(0))
            library_node.add_out_connector("__out_" + field)
            subset = ctx.make_output_dace_subset(node, field)
            state.add_edge(
                library_node,
                "__out_" + field,
                access_node,
                None,
                dace.Memlet(field, subset=subset),
            )

        return

    def visit_Stencil(self, node: oir.Stencil, **kwargs):

        ctx = OirSDFGBuilder.SDFGContext(
            stencil=node,
        )
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
                    debuginfo=dace.DebugInfo(0),
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
                debuginfo=dace.DebugInfo(0),
            )
        self.generic_visit(node, ctx=ctx)
        ctx.sdfg.validate()
        return ctx.sdfg
