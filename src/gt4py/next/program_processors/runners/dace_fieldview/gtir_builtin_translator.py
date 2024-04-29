# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


from typing import Sequence

import dace

from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners.dace_fieldview.gtir_tasklet_codegen import (
    GtirTaskletCodegen,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type
from gt4py.next.type_system import type_specifications as ts


class GtirBuiltinScalarAccess(GtirTaskletCodegen):
    _sym_name: str
    _data_type: ts.ScalarType

    def __init__(
        self, sdfg: dace.SDFG, state: dace.SDFGState, sym_name: str, data_type: ts.ScalarType
    ):
        super().__init__(sdfg, state)
        self._sym_name = sym_name
        self._data_type = data_type

    def __call__(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        tasklet_node = self._state.add_tasklet(
            f"get_{self._sym_name}",
            {},
            {"__out"},
            f"__out = {self._sym_name}",
        )
        name = f"{self._state.label}_var"
        dtype = as_dace_type(self._data_type)
        output_node = self._state.add_scalar(name, dtype, find_new_name=True, transient=True)
        self._state.add_edge(
            tasklet_node, "__out", output_node, None, dace.Memlet(data=output_node.data, subset="0")
        )
        return [(output_node, self._data_type)]


class GtirBuiltinAsFieldOp(GtirTaskletCodegen):
    _stencil: itir.Lambda
    _domain: dict[Dimension, tuple[str, str]]
    _args: Sequence[GtirTaskletCodegen]
    _field_type: ts.FieldType

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        stencil: itir.Lambda,
        domain: Sequence[tuple[Dimension, str, str]],
        args: Sequence[GtirTaskletCodegen],
        field_dtype: ts.ScalarType,
    ):
        super().__init__(sdfg, state)
        self._stencil = stencil
        self._args = args
        self._domain = {dim: (lb, ub) for dim, lb, ub in domain}
        self._field_type = ts.FieldType([dim for dim, _, _ in domain], field_dtype)

    def __call__(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        # generate the python code for this stencil
        output_connector = "__out"
        tlet_code = "{var} = {code}".format(
            var=output_connector, code=self.visit(self._stencil.expr)
        )

        # allocate local (aka transient) storage for the field
        field_shape = [
            # diff between upper and lower bound
            f"{self._domain[dim][1]} - {self._domain[dim][0]}"
            for dim in self._field_type.dims
        ]
        field_node = self._add_local_storage(self._field_type, field_shape)

        # create map range corresponding to the field operator domain
        map_ranges = {f"i_{dim.value}": f"{lb}:{ub}" for dim, (lb, ub) in self._domain.items()}

        # visit expressions passed as arguments to this stencil
        input_nodes: dict[str, dace.nodes.AccessNode] = {}
        input_memlets: dict[str, dace.Memlet] = {}
        assert len(self._args) == len(self._stencil.params)
        for arg, param in zip(self._args, self._stencil.params):
            arg_nodes = arg()
            assert len(arg_nodes) == 1
            arg_node, arg_type = arg_nodes[0]
            connector = str(param.id)
            # require (for now) all input nodes to be data access nodes
            assert isinstance(arg_node, dace.nodes.AccessNode)
            input_nodes[arg_node.data] = arg_node
            if isinstance(arg_type, ts.FieldType):
                # support either single element access (general case) or full array shape
                is_scalar = all(dim in self._domain for dim in arg_type.dims)
                if is_scalar:
                    subset = ",".join(f"i_{dim.value}" for dim in arg_type.dims)
                    input_memlets[connector] = dace.Memlet(
                        data=arg_node.data, subset=subset, volume=1
                    )
                else:
                    memlet = dace.Memlet.from_array(arg_node.data, arg_node.desc(self._sdfg))
                    memlet.volume = 1
                    input_memlets[connector] = memlet
            else:
                input_memlets[connector] = dace.Memlet(data=arg_node.data, subset="0")

        # assume tasklet with single output
        output_index = ",".join(f"i_{dim.value}" for dim in self._field_type.dims)
        output_memlets = {output_connector: dace.Memlet(data=field_node.data, subset=output_index)}
        output_nodes = {field_node.data: field_node}

        # create a tasklet inside a parallel-map scope
        self._state.add_mapped_tasklet(
            "tasklet",
            map_ranges,
            input_memlets,
            tlet_code,
            output_memlets,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            external_edges=True,
        )

        return [(field_node, self._field_type)]


class GtirBuiltinSelect(GtirTaskletCodegen):
    _true_br_builder: GtirTaskletCodegen
    _false_br_builder: GtirTaskletCodegen

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        true_br_builder: GtirTaskletCodegen,
        false_br_builder: GtirTaskletCodegen,
    ):
        super().__init__(sdfg, state)
        self._true_br_builder = true_br_builder
        self._false_br_builder = false_br_builder

    def __call__(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        true_br_args = self._true_br_builder()
        false_br_args = self._false_br_builder()
        assert len(true_br_args) == len(false_br_args)

        output_nodes = []
        for true_br, false_br in zip(true_br_args, false_br_args):
            true_br_node, true_br_type = true_br
            assert isinstance(true_br_node, dace.nodes.AccessNode)
            false_br_node, false_br_type = false_br
            assert isinstance(false_br_node, dace.nodes.AccessNode)
            assert true_br_type == false_br_type
            array_type = self._sdfg.arrays[true_br_node.data]
            access_node = self._add_local_storage(true_br_type, array_type.shape)
            output_nodes.append((access_node, true_br_type))

            data_name = access_node.data
            true_br_output_node = self._true_br_builder._state.add_access(data_name)
            self._true_br_builder._state.add_nedge(
                true_br_node,
                true_br_output_node,
                dace.Memlet.from_array(
                    true_br_output_node.data, true_br_output_node.desc(self._sdfg)
                ),
            )

            false_br_output_node = self._false_br_builder._state.add_access(data_name)
            self._false_br_builder._state.add_nedge(
                false_br_node,
                false_br_output_node,
                dace.Memlet.from_array(
                    false_br_output_node.data, false_br_output_node.desc(self._sdfg)
                ),
            )
        return output_nodes
