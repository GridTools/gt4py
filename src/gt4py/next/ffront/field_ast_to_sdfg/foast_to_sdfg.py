import copy

import dace

import gt4py.eve as eve
from gt4py.next.ffront import field_operator_ast as foast
from gt4py.next.type_system import type_specifications as ts
from .utility import type_spec_to_dtype
from typing import Optional


class FoastToSDFG(eve.NodeTranslator):
    sdfg: dace.SDFG
    last_state: dace.SDFGState

    def visit_Symbol(self, node: foast.Symbol) -> str:
        if isinstance(node.type, ts.FieldType):
            num_dims = len(node.type.dims)
            shape = tuple(
                dace.symbol(f"size{i}", dtype=dace.int64) for i in range(num_dims)
            )
            dtype = type_spec_to_dtype(node.type.dtype)
            self.sdfg.add_array(name=str(node.id), shape=shape, dtype=dtype, transient=False)
            return str(node.id)
        elif isinstance(node.type, ts.ScalarType):
            raise NotImplementedError("rest of the ops don't support scalar arguments")

    def visit_Name(self, node: foast.Name) -> str:
        return str(node.id)

    def visit_Constant(self, node: foast.Constant) -> str:
        raise NotImplementedError("this we don't need")

    def visit_BinOp(self, node: foast.BinOp) -> str:
        assert isinstance(node.type, ts.FieldType)
        num_dims = len(node.type.dims)
        left_name = self.visit(node.left)
        right_name = self.visit(node.right)
        output_name = self.sdfg.temp_data_name()

        shape = tuple(
            dace.symbol(f"size{i}", dtype=dace.int64) for i in range(num_dims)
        )
        dtype = type_spec_to_dtype(node.type.dtype)
        self.sdfg.add_array(name=output_name, shape=shape, dtype=dtype, transient=True)

        self.last_state = self.sdfg.add_state_after(self.last_state, f"binary_op_{node.op}")

        domain = {f"idx{i}": f"0:size{i}" for i in range(num_dims)}

        input_memlets = {
            "left_element": dace.Memlet(
                data=left_name,
                subset=", ".join(f"idx{i}" for i in range(num_dims))
            ),
            "right_element": dace.Memlet(
                data=right_name,
                subset=", ".join(f"idx{i}" for i in range(num_dims))
            ),
        }

        output_memlets = {
            "output_element": dace.Memlet(
                data=output_name,
                subset=", ".join(f"idx{i}" for i in range(num_dims))
            ),
        }

        self.last_state.add_mapped_tasklet(
            name="addition",
            map_ranges=domain,
            inputs=input_memlets,
            code=f"output_element = left_element {node.op} right_element",
            outputs=output_memlets,
            external_edges=True,
            schedule=dace.ScheduleType.Sequential,
        )

        return output_name

    def visit_Return(self, node: foast.Return) -> str:
        return_name = self.visit(node.value)
        return return_name

    def visit_FieldOperator(self, node: foast.FieldOperator) -> str:
        self.sdfg = dace.SDFG(name=node.id)
        self.last_state = self.sdfg.add_state("state", True)

        func = node.definition

        for param in func.params:
            self.visit(param)

        return_name: Optional[str] = None
        for stmt in func.body.stmts:
            return_name: str = self.visit(stmt)
        assert return_name

        return return_name