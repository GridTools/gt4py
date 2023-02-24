# GT4Py - GridTools Framework
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

from typing import Any

import dace

import gt4py.eve as eve
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past
from gt4py.next.type_system import type_specifications as ts

from .foast_to_sdfg import FoastToSDFG
from .utility import type_spec_to_dtype


class PastToSDFG(eve.NodeVisitor):
    sdfg: dace.SDFG
    last_state: dace.SDFGState
    closure_vars: dict[str, Any]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    def visit_Symbol(self, node: past.Symbol) -> str:
        if isinstance(node.type, ts.FieldType):
            num_dims = len(node.type.dims)
            shape = tuple(dace.symbol(f"size{i}", dtype=dace.int64) for i in range(num_dims))
            dtype = type_spec_to_dtype(node.type.dtype)
            self.sdfg.add_array(name=str(node.id), shape=shape, dtype=dtype, transient=False)
            return str(node.id)
        elif isinstance(node.type, ts.ScalarType):
            raise NotImplementedError("rest of the ops don't support scalar arguments")
        raise NotImplementedError()

    def visit_Name(self, node: past.Name) -> str:
        return str(node.id)

    def visit_Call(self, node: past.Call) -> None:
        callee = node.func.id
        out_node = node.kwargs["out"]
        args = set(self.visit(arg) for arg in node.args)
        out = self.visit(out_node)

        foast_node: foast.FieldOperator = self.closure_vars[callee].foast_node
        param_names = [str(param.id) for param in foast_node.definition.params]
        translator = FoastToSDFG()
        call_result_name = translator.visit(foast_node)
        callee_sdfg = translator.sdfg

        self.last_state = self.sdfg.add_state_after(self.last_state)
        nsdfg_node = self.last_state.add_nested_sdfg(
            callee_sdfg, parent=None, inputs=param_names, outputs={call_result_name}
        )
        input_accesses = [self.last_state.add_access(arg) for arg in args]
        output_access = self.last_state.add_access(out)
        assert isinstance(out_node.type, ts.FieldType)
        num_dims = len(out_node.type.dims)

        for inner_name, access_node in zip(param_names, input_accesses):
            self.last_state.add_edge(
                access_node,
                None,
                nsdfg_node,
                inner_name,
                dace.Memlet(
                    data=access_node.data, subset=", ".join(f"0:size{i}" for i in range(num_dims))
                ),
            )

        self.last_state.add_edge(
            nsdfg_node,
            call_result_name,
            output_access,
            None,
            dace.Memlet(
                data=output_access.data, subset=", ".join(f"0:size{i}" for i in range(num_dims))
            ),
        )

        pass

    def visit_Constant(self, node: past.Constant):
        ...

    def visit_Program(self, node: past.Program) -> None:
        self.sdfg = dace.SDFG(name=node.id)
        self.last_state = self.sdfg.add_state("state", True)

        for param in node.params:
            self.visit(param)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Program_that_adds_stuff(self, node: past.Program):
        self.sdfg = dace.SDFG(name=node.id)
        last_state = self.sdfg.add_state("state", True)

        for param in node.params:
            if isinstance(param.type, ts.ScalarType):
                dtype = type_spec_to_dtype(param.type)
                self.sdfg.add_symbol(str(param.id), stype=dtype)
            elif isinstance(param.type, ts.FieldType):
                dtype = type_spec_to_dtype(param.type.dtype)
                shape = tuple(
                    dace.symbol(f"_size_{param.id}_{i}", dtype=dace.int64)
                    for i in range(len(param.type.dims))
                )
                strides = tuple(
                    dace.symbol(f"_stride_{param.id}_{i}", dtype=dace.int64)
                    for i in range(len(param.type.dims))
                )
                self.sdfg.add_array(str(param.id), shape=shape, strides=strides, dtype=dtype)
            else:
                raise ValueError(f"parameter of type {param.type} is not supported")

        out_array = self.sdfg.arrays[node.params[-1].id]
        out_size = out_array.shape
        domain = {f"i{dim}": f"0:{size}" for dim, size in enumerate(out_size)}
        input_memlets = {
            f"{param.id}_element": dace.Memlet(
                data=str(param.id), subset=",".join(f"i{dim}" for dim in range(len(out_size)))
            )
            for param in node.params[0:-1]
        }
        output_memlets = {
            f"{node.params[-1].id}_element": dace.Memlet(
                data=str(node.params[-1].id),
                subset=",".join(f"i{dim}" for dim in range(len(out_size))),
            )
        }

        last_state.add_mapped_tasklet(
            name="addition",
            map_ranges=domain,
            inputs=input_memlets,
            code=f"{node.params[-1].id}_element = {' + '.join(input_memlets.keys())}",
            outputs=output_memlets,
            external_edges=True,
            schedule=dace.ScheduleType.Sequential,
        )
