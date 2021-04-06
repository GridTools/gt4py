# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gtc_unstructured.irs import nir, usid


class NirToUsid(eve.NodeTranslator):
    @dataclass
    class KernelContext:
        primary_composite_name: str
        primary_composite_entries: Set = field(default_factory=set)
        composites: Dict = field(default_factory=dict)

        def add_primary_entry(
            self, entry: Union[usid.SidCompositeEntry, usid.SidCompositeSparseEntry]
        ):
            self.primary_composite_entries.add(entry)
            return self

        def add_sid(self, name: str):
            if name not in self.composites:
                self.composites[name] = set()
            return self

        def add_entry(
            self, name: str, entry: Union[usid.SidCompositeEntry, usid.SidCompositeSparseEntry]
        ) -> "NirToUsid.KernelContext":
            assert name in self.composites
            self.composites[name].add(entry)
            return self

    @dataclass
    class NeighborLoopContext:
        local_index: Optional[str] = None

        def needs_index(self) -> str:
            self.local_index = "i"
            return self.local_index

    def convert_dimensions(self, dims: nir.Dimensions):
        dimensions = []
        if dims.horizontal:
            horizontal = dims.horizontal
            dimensions.append(horizontal.primary)
            # TODO if horizontal.secondary:
            # TODO    dimensions.append(self.visit(horizontal.secondary))
        if dims.vertical:
            dimensions.append(self.visit(dims.vertical))
        return dimensions

    def visit_VerticalDimension(self, node: nir.VerticalDimension, **kwargs):
        return usid.VerticalDimension()

    def visit_UField(self, node: nir.UField, **kwargs):
        return usid.UField(
            name=node.name, vtype=node.vtype, dimensions=self.convert_dimensions(node.dimensions)
        )

    def visit_SparseField(self, node: nir.SparseField, **kwargs):
        return usid.SparseField(
            name=node.name,
            vtype=node.vtype,
            dimensions=self.convert_dimensions(node.dimensions),
            connectivity=node.connectivity,
        )

    def visit_TemporaryField(self, node: nir.TemporaryField, **kwargs):
        return usid.Temporary(
            name=node.name, vtype=node.vtype, dimensions=self.convert_dimensions(node.dimensions)
        )

    def visit_BinaryOp(self, node: nir.BinaryOp, **kwargs):
        return usid.BinaryOp(
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            op=node.op,
            location_type=node.location_type,
        )

    def visit_Literal(self, node: nir.Literal, **kwargs):
        return usid.Literal(value=node.value, vtype=node.vtype, location_type=node.location_type)

    def visit_NeighborLoop(self, node: nir.NeighborLoop, kernel_ctx: "KernelContext", **kwargs):
        neigh_loop_ctx = self.NeighborLoopContext()

        primary_sid = kernel_ctx.primary_composite_name
        secondary_sid = kwargs["symtable"][node.connectivity].name + "_sid"  # node.name
        kernel_ctx.add_sid(secondary_sid)

        primary = "p"
        secondary = "n"

        acc_mapping = {primary_sid: primary, node.name.name: secondary}
        sid_mapping = {primary_sid: primary_sid, node.name.name: secondary_sid}

        kernel_ctx.add_primary_entry(
            usid.SidCompositeEntry(
                ref=node.connectivity, name=f"{node.connectivity}{usid.TAG_APPENDIX}"
            )
        )
        body = self.visit(
            node.body,
            kernel_ctx=kernel_ctx,
            neigh_loop_ctx=neigh_loop_ctx,
            acc_mapping=lambda x: acc_mapping[x],
            sid_mapping=lambda x: sid_mapping[x],
            **kwargs,
        )

        return usid.NeighborLoop(
            primary_sid=kernel_ctx.primary_composite_name,
            secondary_sid=secondary_sid,
            connectivity=node.connectivity,
            primary=usid.PtrRef(name=primary),
            secondary=usid.PtrRef(name=secondary),
            location_type=node.location_type,
            local_index=usid.LocalIndex(name=neigh_loop_ctx.local_index)
            if neigh_loop_ctx.local_index
            else None,
            body=body,
        )

    def visit_FieldAccess(
        self,
        node: nir.FieldAccess,
        *,
        kernel_ctx: "KernelContext",
        acc_mapping=lambda x: x,
        sid_mapping=lambda x: x,
        **kwargs,
    ):
        symtable = kwargs["symtable"]
        field_deref = symtable[node.name]
        sid = sid_mapping(node.primary)
        ref = acc_mapping(node.primary)
        name = f"{node.name}_{sid}"
        if isinstance(field_deref, nir.SparseField):
            kernel_ctx.add_primary_entry(
                usid.SidCompositeSparseEntry(
                    ref=node.name, name=name, connectivity=field_deref.connectivity
                )
            )
        elif isinstance(field_deref, nir.LocalFieldVar):
            assert "neigh_loop_ctx" in kwargs
            neigh_loop_ctx = kwargs["neigh_loop_ctx"]
            return usid.ArrayAccess(
                name=node.name,
                subscript=neigh_loop_ctx.needs_index(),
                location_type=node.location_type,
            )
        else:
            if sid == kernel_ctx.primary_composite_name:
                kernel_ctx.add_primary_entry(usid.SidCompositeEntry(ref=node.name, name=name))
            else:
                kernel_ctx.add_entry(sid, usid.SidCompositeEntry(ref=node.name, name=name))

        return usid.FieldAccess(
            name=name,
            sid=ref,
            location_type=node.location_type,
        )

    def visit_VarAccess(self, node: nir.VarAccess, **kwargs):
        return usid.VarAccess(name=node.name, location_type=node.location_type)

    def visit_AssignStmt(self, node: nir.AssignStmt, **kwargs):
        return usid.AssignStmt(
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            location_type=node.location_type,
        )

    def visit_NativeFuncCall(self, node: nir.NativeFuncCall, **kwargs):
        return usid.NativeFuncCall(
            func=node.func,
            args=self.visit(node.args, **kwargs),
            location_type=node.location_type,
        )

    def visit_LocalVar(self, node: nir.LocalVar, **kwargs):
        return usid.VarDecl(
            name=node.name,
            init=usid.Literal(value="0.0", vtype=node.vtype, location_type=node.location_type),
            vtype=node.vtype,
            location_type=node.location_type,
        )

    def visit_LocalFieldVar(self, node: nir.LocalFieldVar, *, symtable, **kwargs):
        connectivity_deref: nir.Connectivity = symtable[node.connectivity]
        return usid.StaticArrayDecl(
            name=node.name,
            init=self.visit(node.init),
            vtype=node.init[
                0
            ].vtype,  # TODO bad hack, need to check that all are the same and can be deduced
            size=connectivity_deref.max_neighbors,
            location_type=node.location_type,
        )

    def visit_BlockStmt(self, node: nir.BlockStmt, **kwargs):
        statements = self.visit(node.declarations, **kwargs)
        for stmt in node.statements:
            statements.append(self.visit(stmt, **kwargs))
        return statements

    def visit_HorizontalLoop(self, node: nir.HorizontalLoop, *, symtable, **kwargs):

        kernel_ctx = self.KernelContext(node.iteration_space.name)
        body = self.visit(
            node.stmt, kernel_ctx=kernel_ctx, symtable={**symtable, **node.symtable_}, **kwargs
        )
        primary_composite = usid.SidComposite(
            name=node.iteration_space.name, entries=kernel_ctx.primary_composite_entries
        )
        composites = kernel_ctx.composites
        secondary_composites = [
            usid.SidComposite(name=name, entries=entries) for name, entries in composites.items()
        ]

        kernel_name = "kernel_" + node.id_
        debug(primary_composite)
        debug(secondary_composites)
        kernel = usid.Kernel(
            name=kernel_name,
            primary_location=node.iteration_space.location_type,
            primary_composite=primary_composite,
            secondary_composites=secondary_composites,
            body=body,
        )
        return kernel, usid.KernelCall(name=kernel_name)

    def visit_VerticalLoop(self, node: nir.VerticalLoop, **kwargs):
        # TODO I am completely ignoring k loops at this point!
        kernels = []
        kernel_calls = []
        for loop in node.horizontal_loops:
            k, c = self.visit(loop, **kwargs)
            kernels.append(k)
            kernel_calls.append(c)
        return kernels, kernel_calls

    def visit_Stencil(self, node: nir.Stencil, **kwargs):
        kernels = []
        kernel_calls = []
        for loop in node.vertical_loops:
            k, c = self.visit(loop, **kwargs)
            kernels.extend(k)
            kernel_calls.extend(c)
        return kernels, kernel_calls

    def visit_Connectivity(self, node: nir.Connectivity, **kwargs):
        return usid.Connectivity(
            name=node.name,
            max_neighbors=node.max_neighbors,
            has_skip_values=node.has_skip_values,
        )

    def visit_Computation(self, node: nir.Computation, **kwargs):
        parameters = []
        for f in node.params:  # before visiting stencils!
            converted_param = self.visit(f)
            parameters.append(converted_param)

        temporaries = []
        for tmp in node.declarations or []:
            converted_tmp = self.visit(tmp)
            temporaries.append(converted_tmp)

        kernels = []
        ctrlflow_ast = []
        for s in node.stencils:
            kernel, kernel_call = self.visit(s, symtable=node.symtable_)
            kernels.extend(kernel)
            ctrlflow_ast.extend(kernel_call)

        return usid.Computation(
            name=node.name,
            connectivities=self.visit(node.connectivities),
            parameters=parameters,
            temporaries=temporaries,
            kernels=kernels,
            ctrlflow_ast=ctrlflow_ast,
        )
