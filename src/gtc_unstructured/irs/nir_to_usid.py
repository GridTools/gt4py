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
from typing import Dict, Set, Union
from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gtc_unstructured.irs import common, nir, usid


def location_type_from_dimensions(dimensions):
    location_type = [dim for dim in dimensions if isinstance(dim, common.LocationType)]
    if len(location_type) == 1:
        return location_type[0]
    elif len(location_type) == 0:
        return None
    else:
        raise ValueError("Invalid!")


class NirToUsid(eve.NodeTranslator):
    def __init__(self, **kwargs):
        super().__init__()
        self.fields = dict()  # poor man symbol table

    @dataclass
    class KernelContext:
        """"""

        primary_composite_name: str
        primary_composite_entries: Set = field(default_factory=set)
        composites: Dict = field(default_factory=dict)

        def init(self, primary_composite_name: str) -> None:
            self.primary_composite_name = primary_composite_name

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
            # if name not in self.composites:
            #     self.composites[name] = set()
            self.composites[name].add(entry)
            return self

    def convert_dimensions(self, dims: nir.Dimensions):
        dimensions = []
        if dims.horizontal:
            horizontal = dims.horizontal
            dimensions.append(horizontal.primary)
            # if horizontal.secondary:
            #     dimensions.append(self.visit(horizontal.secondary))
        if dims.vertical:
            dimensions.append(self.visit(dims.vertical))
        return dimensions

    # def visit_NeighborChain(self, node: nir.NeighborChain, **kwargs):
    #     return usid.NeighborChain(elements=[location for location in node.elements])

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
        primary_sid = kernel_ctx.primary_composite_name
        secondary_sid = node.name
        kernel_ctx.add_sid(secondary_sid)

        primary = "p"
        secondary = "n"

        acc_mapping = {primary_sid: primary, secondary_sid: secondary}

        kernel_ctx.add_primary_entry(usid.SidCompositeEntry(ref=node.connectivity))

        return usid.NeighborLoop(
            primary_sid=kernel_ctx.primary_composite_name,
            secondary_sid=secondary_sid,
            connectivity=node.connectivity,
            primary=usid.PtrRef(name=primary),
            secondary=usid.PtrRef(name=secondary),
            location_type=node.location_type,
            body=self.visit(node.body, kernel_ctx=kernel_ctx, acc_mapping=acc_mapping, **kwargs),
        )

    def visit_FieldAccess(self, node: nir.FieldAccess, kernel_ctx: "KernelContext", **kwargs):
        symtable = kwargs["symtable"]
        field_deref = symtable[node.name]
        location_deref = symtable[node.primary]
        if "acc_mapping" not in kwargs:
            sid = location_deref.name
            kernel_ctx.add_primary_entry(usid.SidCompositeEntry(ref=node.name))
            ref = sid
        else:
            assert "acc_mapping" in kwargs
            acc_mapping: Dict[str, str] = kwargs["acc_mapping"]
            sid = node.primary
            ref = acc_mapping[sid]
            debug(sid)
            debug(acc_mapping)
            debug(ref)
            if isinstance(field_deref, nir.SparseField):
                kernel_ctx.add_primary_entry(
                    usid.SidCompositeSparseEntry(
                        ref=node.name, connectivity=field_deref.connectivity
                    )
                )
                ref = "p"  # todo
            else:
                if sid == kernel_ctx.primary_composite_name:
                    kernel_ctx.add_primary_entry(usid.SidCompositeEntry(ref=node.name))
                else:
                    kernel_ctx.add_entry(sid, usid.SidCompositeEntry(ref=node.name))

        return usid.FieldAccess(
            name=node.name + "_tag",
            sid=ref,  # kwargs["sids_tbl"][self.visit(node.primary, **kwargs)].name,
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

    def visit_BlockStmt(self, node: nir.BlockStmt, **kwargs):
        statements = []
        for decl in node.declarations:
            statements.append(
                usid.VarDecl(
                    name=decl.name,
                    init=usid.Literal(
                        value="0.0", vtype=decl.vtype, location_type=node.location_type
                    ),
                    vtype=decl.vtype,
                    location_type=node.location_type,
                )
            )
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
            self.fields[converted_param.name] = converted_param

        temporaries = []
        for tmp in node.declarations or []:
            converted_tmp = self.visit(tmp)
            temporaries.append(converted_tmp)
            self.fields[converted_tmp.name] = converted_tmp

        kernels = []
        ctrlflow_ast = []
        for s in node.stencils:
            kernel, kernel_call = self.visit(s, symtable=node.symtable_)
            kernels.extend(kernel)
            ctrlflow_ast.extend(kernel_call)

        debug(kernels)

        return usid.Computation(
            name=node.name,
            connectivities=self.visit(node.connectivities),
            parameters=parameters,
            temporaries=temporaries,
            kernels=kernels,
            ctrlflow_ast=ctrlflow_ast,
        )
