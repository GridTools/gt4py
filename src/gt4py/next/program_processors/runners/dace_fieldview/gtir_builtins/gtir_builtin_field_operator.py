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


from typing import TypeAlias

import dace
import dace.subsets as sbs

from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtins.gtir_builtin_translator import (
    GTIRBuiltinTranslator,
    SDFGField,
    SDFGFieldBuilder,
)
from gt4py.next.program_processors.runners.dace_fieldview.gtir_to_tasklet import (
    GTIRToTasklet,
    IteratorExpr,
    MemletExpr,
    SymbolExpr,
    ValueExpr,
)
from gt4py.next.type_system import type_specifications as ts


# Define type of variables used for field indexing
_INDEX_DTYPE = dace.int64


class GTIRBuiltinAsFieldOp(GTIRBuiltinTranslator):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    TaskletConnector: TypeAlias = tuple[dace.nodes.Tasklet, str]

    stencil_expr: itir.Lambda
    stencil_args: list[SDFGFieldBuilder]
    field_domain: dict[Dimension, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]
    field_type: ts.FieldType
    offset_provider: dict[str, Connectivity | Dimension]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        node: itir.FunCall,
        stencil_args: list[SDFGFieldBuilder],
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        super().__init__(sdfg, state)
        self.offset_provider = offset_provider

        assert cpm.is_call_to(node.fun, "as_fieldop")
        assert len(node.fun.args) == 2
        stencil_expr, domain_expr = node.fun.args
        # expect stencil (represented as a lambda function) as first argument
        assert isinstance(stencil_expr, itir.Lambda)
        # the domain of the field operator is passed as second argument
        assert isinstance(domain_expr, itir.FunCall)

        domain = dace_fieldview_util.get_domain(domain_expr)
        # define field domain with all dimensions in alphabetical order
        sorted_domain_dims = sorted(domain.keys(), key=lambda x: x.value)

        # add local storage to compute the field operator over the given domain
        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        self.field_domain = domain
        self.field_type = ts.FieldType(sorted_domain_dims, node_type)
        self.stencil_expr = stencil_expr
        self.stencil_args = stencil_args

    def build(self) -> list[SDFGField]:
        dimension_index_fmt = "i_{dim}"
        # first visit the list of arguments and build a symbol map
        stencil_args: list[IteratorExpr | MemletExpr] = []
        for arg in self.stencil_args:
            arg_nodes = arg()
            assert len(arg_nodes) == 1
            data_node, arg_type = arg_nodes[0]
            # require all argument nodes to be data access nodes (no symbols)
            assert isinstance(data_node, dace.nodes.AccessNode)

            if isinstance(arg_type, ts.ScalarType):
                scalar_arg = MemletExpr(data_node, sbs.Indices([0]))
                stencil_args.append(scalar_arg)
            else:
                assert isinstance(arg_type, ts.FieldType)
                indices: dict[str, MemletExpr | SymbolExpr | ValueExpr] = {
                    dim.value: SymbolExpr(
                        dace.symbolic.SymExpr(dimension_index_fmt.format(dim=dim.value)),
                        _INDEX_DTYPE,
                    )
                    for dim in self.field_domain.keys()
                }
                iterator_arg = IteratorExpr(
                    data_node,
                    [dim.value for dim in arg_type.dims],
                    indices,
                )
                stencil_args.append(iterator_arg)

        # represent the field operator as a mapped tasklet graph, which will range over the field domain
        taskgen = GTIRToTasklet(self.sdfg, self.head_state, self.offset_provider)
        input_connections, output_expr = taskgen.visit(self.stencil_expr, args=stencil_args)
        assert isinstance(output_expr, ValueExpr)

        # retrieve the tasklet node which writes the result
        output_tasklet_node = self.head_state.in_edges(output_expr.node)[0].src
        output_tasklet_connector = self.head_state.in_edges(output_expr.node)[0].src_conn

        # the last transient node can be deleted
        self.head_state.remove_node(output_expr.node)

        # allocate local temporary storage for the result field
        field_shape = [
            # diff between upper and lower bound
            self.field_domain[dim][1] - self.field_domain[dim][0]
            for dim in self.field_type.dims
        ]
        field_node = self.add_local_storage(self.field_type, field_shape)

        # assume tasklet with single output
        output_index = ",".join(
            dimension_index_fmt.format(dim=dim.value) for dim in self.field_type.dims
        )
        output_memlet = dace.Memlet(data=field_node.data, subset=output_index)

        # create map range corresponding to the field operator domain
        map_ranges = {
            dimension_index_fmt.format(dim=dim.value): f"{lb}:{ub}"
            for dim, (lb, ub) in self.field_domain.items()
        }
        me, mx = self.head_state.add_map("field_op", map_ranges)

        for data_node, data_subset, lambda_node, lambda_connector in input_connections:
            memlet = dace.Memlet(data=data_node.data, subset=data_subset)
            self.head_state.add_memlet_path(
                data_node,
                me,
                lambda_node,
                dst_conn=lambda_connector,
                memlet=memlet,
            )
        self.head_state.add_memlet_path(
            output_tasklet_node,
            mx,
            field_node,
            src_conn=output_tasklet_connector,
            memlet=output_memlet,
        )

        return [(field_node, self.field_type)]
