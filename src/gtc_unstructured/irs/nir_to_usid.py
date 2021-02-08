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

    def convert_dimensions(self, dims: nir.Dimensions):
        dimensions = []
        if dims.horizontal:
            horizontal = dims.horizontal
            dimensions.append(horizontal.primary)
            if horizontal.secondary:
                dimensions.append(self.visit(horizontal.secondary))
        if dims.vertical:
            dimensions.append(self.visit(dims.vertical))
        return dimensions

    def visit_NeighborChain(self, node: nir.NeighborChain, **kwargs):
        return usid.NeighborChain(elements=[location for location in node.elements])

    def visit_VerticalDimension(self, node: nir.VerticalDimension, **kwargs):
        return usid.VerticalDimension()

    def visit_UField(self, node: nir.UField, **kwargs):
        return usid.UField(
            name=node.name, vtype=node.vtype, dimensions=self.convert_dimensions(node.dimensions)
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

    def visit_NeighborLoop(self, node: nir.NeighborLoop, **kwargs):
        return usid.NeighborLoop(
            outer_sid=kwargs["sids_tbl"][usid.NeighborChain(elements=[node.location_type])].name,
            connectivity=kwargs["conn_tbl"][node.neighbors].name,
            sid=kwargs["sids_tbl"][node.neighbors].name
            if node.neighbors in kwargs["sids_tbl"]
            else None,
            location_type=node.location_type,
            body_location_type=node.neighbors.elements[-1],
            body=self.visit(node.body, **kwargs),
        )

    def visit_FieldAccess(self, node: nir.FieldAccess, **kwargs):
        return usid.FieldAccess(
            name=node.name,
            sid=kwargs["sids_tbl"][self.visit(node.primary, **kwargs)].name,
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

    def visit_HorizontalLoop(self, node: nir.HorizontalLoop, **kwargs):
        location_type_str = str(common.LocationType(node.location_type).name).lower()
        primary_connectivity = location_type_str + "_conn"
        connectivities = set()
        connectivities.add(
            usid.Connectivity(
                name=primary_connectivity, chain=usid.NeighborChain(elements=[node.location_type])
            )
        )

        field_accesses = eve.iter_tree(node.stmt).if_isinstance(nir.FieldAccess).to_list()

        other_sids_entries = {}
        primary_sid_entries = set()
        for acc in field_accesses:
            if len(acc.primary.elements) == 1:
                assert acc.primary.elements[0] == node.location_type
                primary_sid_entries.add(usid.SidCompositeEntry(name=acc.name))
            else:
                assert (
                    len(acc.primary.elements) == 2
                )  # TODO cannot deal with more than one level of nesting
                secondary_loc = acc.primary.elements[
                    -1
                ]  # TODO change if we have more than one level of nesting
                if secondary_loc not in other_sids_entries:
                    other_sids_entries[secondary_loc] = set()
                other_sids_entries[secondary_loc].add(usid.SidCompositeEntry(name=acc.name))

        neighloops = eve.iter_tree(node.stmt).if_isinstance(nir.NeighborLoop).to_list()
        for loop in neighloops:
            transformed_neighbors = self.visit(loop.neighbors, **kwargs)
            connectivity_name = str(transformed_neighbors) + "_conn"
            connectivities.add(
                usid.Connectivity(name=connectivity_name, chain=transformed_neighbors)
            )
            primary_sid_entries.add(
                usid.SidCompositeNeighborTableEntry(connectivity=connectivity_name)
            )

        primary_sid = location_type_str
        sids = []
        sids.append(
            usid.SidComposite(
                name=primary_sid,
                entries=primary_sid_entries,
                location=usid.NeighborChain(elements=[node.location_type]),
            )
        )

        for k, v in other_sids_entries.items():
            chain = usid.NeighborChain(elements=[node.location_type, k])
            sids.append(
                usid.SidComposite(name=str(chain), entries=v, location=chain)
            )  # TODO _conn via property

        kernel_name = "kernel_" + node.id_
        kernel = usid.Kernel(
            ast=self.visit(
                node.stmt,
                sids_tbl={s.location: s for s in sids},
                conn_tbl={c.chain: c for c in connectivities},
                **kwargs,
            ),
            name=kernel_name,
            primary_connectivity=primary_connectivity,
            primary_sid=primary_sid,
            connectivities=connectivities,
            sids=sids,
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
            kernel, kernel_call = self.visit(s)
            kernels.extend(kernel)
            ctrlflow_ast.extend(kernel_call)

        debug(kernels)

        return usid.Computation(
            name=node.name,
            parameters=parameters,
            temporaries=temporaries,
            kernels=kernels,
            ctrlflow_ast=ctrlflow_ast,
        )
