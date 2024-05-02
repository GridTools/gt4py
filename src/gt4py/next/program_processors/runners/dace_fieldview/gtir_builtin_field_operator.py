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


from typing import Callable

import dace

from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_dataflow_builder import (
    GtirDataflowBuilder,
)
from gt4py.next.type_system import type_specifications as ts


class GtirBuiltinAsFieldOp(GtirDataflowBuilder):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    _stencil_expr: itir.Lambda
    _stencil_args: list[Callable]
    _field_domain: dict[Dimension, tuple[str, str]]
    _field_type: ts.FieldType

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        data_types: dict[str, ts.FieldType | ts.ScalarType],
        node: itir.FunCall,
        stencil_args: list[Callable],
    ):
        super().__init__(sdfg, state, data_types)

        assert cpm.is_call_to(node.fun, "as_fieldop")
        assert len(node.fun.args) == 2
        stencil_expr, domain_expr = node.fun.args
        # expect stencil (represented as a lambda function) as first argument
        assert isinstance(stencil_expr, itir.Lambda)
        # the domain of the field operator is passed as second argument
        assert isinstance(domain_expr, itir.FunCall)

        # visit field domain
        domain = self.visit_domain(domain_expr)

        # add local storage to compute the field operator over the given domain
        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        self._field_domain = {dim: (lb, ub) for dim, lb, ub in domain}
        self._field_type = ts.FieldType([dim for dim, _, _ in domain], node_type)
        self._stencil_expr = stencil_expr
        self._stencil_args = stencil_args

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        # generate a tasklet node implementing the stencil function and represent
        # the field operator as a mapped tasklet, which will range over the field domain
        output_connector = "__out"
        tlet_code = "{var} = {code}".format(
            var=output_connector, code=self.visit_symbolic(self._stencil_expr.expr)
        )

        # allocate local temporary storage for the result field
        field_shape = [
            # diff between upper and lower bound
            f"{self._field_domain[dim][1]} - {self._field_domain[dim][0]}"
            for dim in self._field_type.dims
        ]
        field_node = self._add_local_storage(self._field_type, field_shape)

        # create map range corresponding to the field operator domain
        map_ranges = {
            f"i_{dim.value}": f"{lb}:{ub}" for dim, (lb, ub) in self._field_domain.items()
        }

        input_nodes: dict[str, dace.nodes.AccessNode] = {}
        input_memlets: dict[str, dace.Memlet] = {}
        assert len(self._stencil_args) == len(self._stencil_expr.params)
        for arg, param in zip(self._stencil_args, self._stencil_expr.params):
            arg_nodes = arg()
            assert len(arg_nodes) == 1
            arg_node, arg_type = arg_nodes[0]
            connector = str(param.id)
            # require (for now) all input nodes to be data access nodes
            assert isinstance(arg_node, dace.nodes.AccessNode)
            input_nodes[arg_node.data] = arg_node
            if isinstance(arg_type, ts.FieldType):
                # support either single element access (general case) or full array shape
                is_scalar = all(dim in self._field_domain for dim in arg_type.dims)
                if is_scalar:
                    subset = ",".join(f"i_{dim.value}" for dim in arg_type.dims)
                    input_memlets[connector] = dace.Memlet(
                        data=arg_node.data, subset=subset, volume=1
                    )
                else:
                    memlet = dace.Memlet.from_array(arg_node.data, arg_node.desc(self._sdfg))
                    # set volume to 1 because the stencil function always performs single element access
                    # TODO: check validity of this assumption
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
