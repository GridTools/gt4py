# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from gtc.dace.nodes import StencilComputation
from gtc.dace.utils import DaceStrMaker, data_type_to_dace_typeclass
from gtc.passes.oir_optimizations.utils import AccessCollector, compute_horizontal_block_extents


class OirSDFGBuilder(eve.NodeVisitor):
    @dataclass
    class Context:
        sdfg: dace.SDFG
        last_state: dace.SDFGState
        dace_str_maker: DaceStrMaker
        declarations: Dict[str, oir.Decl]

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, *, block_extents, ctx: "OirSDFGBuilder.Context", **kwargs
    ):
        declarations = {
            acc.name: ctx.declarations[acc.name]
            for acc in node.iter_tree().if_isinstance(oir.FieldAccess, oir.ScalarAccess)
            if acc.name in ctx.declarations
        }
        library_node = StencilComputation(
            name=f"{ctx.sdfg.name}_computation_{id(node)}",
            extents=block_extents,
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
            subset = ctx.dace_str_maker.make_input_dace_subset(node, field)
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
            subset = ctx.dace_str_maker.make_output_dace_subset(node, field)
            state.add_edge(
                library_node,
                "__out_" + field,
                access_node,
                None,
                dace.Memlet(field, subset=subset),
            )

        return

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        sdfg = dace.SDFG(node.name)
        state = sdfg.add_state(is_start_state=True)

        block_extents = compute_horizontal_block_extents(node)
        ctx = OirSDFGBuilder.Context(
            sdfg=sdfg,
            last_state=state,
            dace_str_maker=DaceStrMaker(node),
            declarations={decl.name: decl for decl in node.params + node.declarations},
        )
        for param in node.params:
            if isinstance(param, oir.FieldDecl):
                dim_strs = [d for i, d in enumerate("IJK") if param.dimensions[i]] + [
                    f"d{d}" for d in range(len(param.data_dims))
                ]
                sdfg.add_array(
                    param.name,
                    shape=ctx.dace_str_maker.make_shape(param.name),
                    strides=[
                        dace.symbolic.pystr_to_symbolic(f"__{param.name}_{dim}_stride")
                        for dim in dim_strs
                    ],
                    dtype=data_type_to_dace_typeclass(param.dtype),
                    transient=False,
                    debuginfo=dace.DebugInfo(0),
                )
            else:
                sdfg.add_symbol(param.name, stype=data_type_to_dace_typeclass(param.dtype))

        for decl in node.declarations:
            dim_strs = [d for i, d in enumerate("IJK") if decl.dimensions[i]] + [
                f"d{d}" for d in range(len(decl.data_dims))
            ]
            sdfg.add_array(
                decl.name,
                shape=ctx.dace_str_maker.make_shape(decl.name),
                strides=[
                    dace.symbolic.pystr_to_symbolic(f"__{decl.name}_{dim}_stride")
                    for dim in dim_strs
                ],
                dtype=data_type_to_dace_typeclass(decl.dtype),
                transient=True,
                debuginfo=dace.DebugInfo(0),
            )
        self.generic_visit(node, ctx=ctx, block_extents=block_extents)
        ctx.sdfg.validate()
        return ctx.sdfg
