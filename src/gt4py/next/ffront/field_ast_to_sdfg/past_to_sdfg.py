from typing import Any

import dace

import gt4py.eve as eve
from gt4py.next.ffront import program_ast as past
from gt4py.next.type_system import type_specifications as ts
from .utility import type_spec_to_dtype
from .foast_to_sdfg import FoastToSDFG


class PastToSDFG(eve.NodeVisitor):
    sdfg: dace.SDFG
    last_state: dace.SDFGState
    closure_vars: dict[str, Any]

    def visit_Symbol(self, node: past.Symbol):
        ...

    def visit_Name(self, node: past.Name):
        ...

    def visit_Call(self, node: past.Call):
        callee = node.func.id
        args = set(self.visit(arg) for arg in node.args)
        assert not node.kwargs

        foast_node = self.closure_vars[callee].foast_node
        translator = FoastToSDFG()
        call_result_name = translator.visit(foast_node)
        callee_sdfg = translator.sdfg

        self.last_state = self.sdfg.add_state_after(self.last_state)
        nsdfg_node = self.last_state.add_nested_sdfg(callee_sdfg, parent=None, inputs=args, outputs={call_result_name})
        input_accesses = {arg: self.last_state.add_access(arg) for arg in args}
        output_access = self.last_state.add_access(call_result_name)





    def visit_Constant(self, node: past.Constant):
        ...

    def visit_Program(self, node: past.Program):
        self.sdfg = dace.SDFG(name=node.id)
        self.last_state = self.sdfg.add_state("state", True)

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
