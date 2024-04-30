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
from gt4py.next.type_system import type_specifications as ts


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

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
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
