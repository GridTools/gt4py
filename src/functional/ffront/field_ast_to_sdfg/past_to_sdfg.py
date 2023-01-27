import dace

import eve
from functional.ffront import program_ast as past
from functional.type_system import type_specifications as ts


def type_spec_to_dtype(type_: ts.ScalarType):
    if type_.kind == ts.ScalarKind.BOOL:
        return dace.bool_
    elif type_.kind == ts.ScalarKind.INT32:
        return dace.int32
    elif type_.kind == ts.ScalarKind.INT64:
        return dace.int64
    elif type_.kind == ts.ScalarKind.FLOAT32:
        return dace.float32
    elif type_.kind == ts.ScalarKind.FLOAT64:
        return dace.float64
    raise ValueError(f"scalar type {type_} not supported")


class PastToSDFG(eve.NodeVisitor):
    sdfg: dace.SDFG

    def visit_Program(self, node: past.Program):
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

        input_arrays = [self.sdfg.arrays[param.id] for param in node.params[0:-1]]
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
