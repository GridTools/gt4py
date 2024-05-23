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


import abc
from dataclasses import dataclass
from typing import Callable, TypeAlias, final

import dace
import dace.subsets as sbs

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_to_tasklet,
    utility as dace_fieldview_util,
)
from gt4py.next.type_system import type_specifications as ts


# Define aliases for return types
SDFGField: TypeAlias = tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]
SDFGFieldBuilder: TypeAlias = Callable[[], list[SDFGField]]

DIMENSION_INDEX_FMT = "i_{dim}"


@dataclass(frozen=True)
class PrimitiveTranslator(abc.ABC):
    sdfg: dace.SDFG
    head_state: dace.SDFGState

    @final
    def __call__(self) -> list[SDFGField]:
        """The callable interface is used to build the dataflow graph.

        It allows to build the dataflow graph inside a given state starting
        from the innermost nodes, by propagating the intermediate results
        as access nodes to temporary local storage.
        """
        return self.build()

    @final
    def add_local_storage(
        self, data_type: ts.FieldType | ts.ScalarType, shape: list[str]
    ) -> dace.nodes.AccessNode:
        """
        Allocates temporary storage to be used in the local scope for intermediate results.

        The storage is allocate with unique names to enable SSA optimization in the compilation phase.
        """
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = dace_fieldview_util.as_dace_type(data_type.dtype)
            name, _ = self.sdfg.add_array("var", shape, dtype, find_new_name=True, transient=True)
        else:
            assert len(shape) == 0
            dtype = dace_fieldview_util.as_dace_type(data_type)
            name, _ = self.sdfg.add_scalar("var", dtype, find_new_name=True, transient=True)
        return self.head_state.add_access(name)

    @abc.abstractmethod
    def build(self) -> list[SDFGField]:
        """Creates the dataflow subgraph representing a GTIR builtin function.

        This method is used by derived classes to build a specialized subgraph
        for a specific builtin function.

        Returns a list of SDFG nodes and the associated GT4Py data type.

        The GT4Py data type is useful in the case of fields, because it provides
        information on the field domain (e.g. order of dimensions, types of dimensions).
        """


class AsFieldOp(PrimitiveTranslator):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    TaskletConnector: TypeAlias = tuple[dace.nodes.Tasklet, str]

    stencil_expr: itir.Lambda
    stencil_args: list[SDFGFieldBuilder]
    field_domain: dict[str, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]
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
        sorted_domain_dims = sorted(
            [Dimension(dim) for dim in domain.keys()], key=lambda x: x.value
        )

        # add local storage to compute the field operator over the given domain
        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        self.field_domain = domain
        self.field_type = ts.FieldType(sorted_domain_dims, node_type)
        self.stencil_expr = stencil_expr
        self.stencil_args = stencil_args

    def build_reduce_node(
        self,
        input_connections: list[gtir_to_tasklet.InputConnection],
        output_expr: gtir_to_tasklet.ValueExpr,
    ) -> list[SDFGField]:
        """Use dace library node for reduction."""

        assert len(input_connections) == 1
        source_node, source_subset, reduce_node, reduce_connector = input_connections[0]
        assert reduce_connector is None

        self.head_state.add_nedge(
            source_node,
            reduce_node,
            dace.Memlet(data=source_node.data, subset=source_subset),
        )

        return [(output_expr.node, self.field_type)]

    def build_tasklet_node(
        self,
        input_connections: list[gtir_to_tasklet.InputConnection],
        output_expr: gtir_to_tasklet.ValueExpr,
    ) -> list[SDFGField]:
        """Translates the field operator to a mapped tasklet graph, which will range over the field domain."""

        # retrieve the tasklet node which writes the result
        output_tasklet_node = self.head_state.in_edges(output_expr.node)[0].src
        output_tasklet_connector = self.head_state.in_edges(output_expr.node)[0].src_conn

        # the last transient node can be deleted
        # TODO: not needed to store the node `dtype` after type inference is in place
        output_desc = output_expr.node.desc(self.sdfg)
        self.head_state.remove_node(output_expr.node)

        # allocate local temporary storage for the result field
        field_dims = self.field_type.dims.copy()
        field_shape = [
            # diff between upper and lower bound
            self.field_domain[dim.value][1] - self.field_domain[dim.value][0]
            for dim in self.field_type.dims
        ]
        if isinstance(output_desc, dace.data.Array):
            # extend the result arrays with the local dimensions added by the field operator e.g. `neighbors`)
            field_dims.extend(Dimension(f"local_dim{i}") for i in range(len(output_desc.shape)))
            field_shape.extend(output_desc.shape)

        # TODO: use `self.field_type` without overriding `dtype` when type inference is in place
        field_dtype = dace_fieldview_util.as_scalar_type(str(output_desc.dtype.as_numpy_dtype()))
        field_node = self.add_local_storage(ts.FieldType(field_dims, field_dtype), field_shape)

        # assume tasklet with single output
        output_subset = [DIMENSION_INDEX_FMT.format(dim=dim.value) for dim in self.field_type.dims]
        if isinstance(output_desc, dace.data.Array):
            output_subset.extend(f"0:{size}" for size in output_desc.shape)
        output_memlet = dace.Memlet(data=field_node.data, subset=",".join(output_subset))

        # create map range corresponding to the field operator domain
        map_ranges = {
            DIMENSION_INDEX_FMT.format(dim=dim): f"{lb}:{ub}"
            for dim, (lb, ub) in self.field_domain.items()
        }
        me, mx = self.head_state.add_map("field_op", map_ranges)

        if len(input_connections) == 0:
            # dace requires an empty edge from map entry node to tasklet node, in case there no input memlets
            self.head_state.add_nedge(me, output_tasklet_node, dace.Memlet())
        else:
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

    def build(self) -> list[SDFGField]:
        # type of variables used for field indexing
        index_dtype = dace.int32
        # first visit the list of arguments and build a symbol map
        stencil_args: list[gtir_to_tasklet.IteratorExpr | gtir_to_tasklet.MemletExpr] = []
        for arg in self.stencil_args:
            arg_nodes = arg()
            assert len(arg_nodes) == 1
            data_node, arg_type = arg_nodes[0]
            # require all argument nodes to be data access nodes (no symbols)
            assert isinstance(data_node, dace.nodes.AccessNode)

            if isinstance(arg_type, ts.ScalarType):
                scalar_arg = gtir_to_tasklet.MemletExpr(data_node, sbs.Indices([0]))
                stencil_args.append(scalar_arg)
            else:
                assert isinstance(arg_type, ts.FieldType)
                indices: dict[str, gtir_to_tasklet.IteratorIndexExpr] = {
                    dim: gtir_to_tasklet.SymbolExpr(
                        dace.symbolic.SymExpr(DIMENSION_INDEX_FMT.format(dim=dim)),
                        index_dtype,
                    )
                    for dim in self.field_domain.keys()
                }
                iterator_arg = gtir_to_tasklet.IteratorExpr(
                    data_node,
                    [dim.value for dim in arg_type.dims],
                    indices,
                )
                stencil_args.append(iterator_arg)

        taskgen = gtir_to_tasklet.LambdaToTasklet(self.sdfg, self.head_state, self.offset_provider)
        input_connections, output_expr = taskgen.visit(self.stencil_expr, args=stencil_args)
        assert isinstance(output_expr, gtir_to_tasklet.ValueExpr)

        if cpm.is_applied_reduce(self.stencil_expr.expr) and cpm.is_call_to(
            self.stencil_expr.expr.args[0], "deref"
        ):
            return self.build_reduce_node(input_connections, output_expr)

        else:
            return self.build_tasklet_node(input_connections, output_expr)


class Select(PrimitiveTranslator):
    """Generates the dataflow subgraph for the `select` builtin function."""

    true_br_builder: SDFGFieldBuilder
    false_br_builder: SDFGFieldBuilder

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        dataflow_builder: eve.NodeVisitor,
        node: itir.FunCall,
    ):
        super().__init__(sdfg, state)

        assert cpm.is_call_to(node.fun, "select")
        assert len(node.fun.args) == 3
        cond_expr, true_expr, false_expr = node.fun.args

        # expect condition as first argument
        cond = dace_fieldview_util.get_symbolic_expr(cond_expr)

        # use current head state to terminate the dataflow, and add a entry state
        # to connect the true/false branch states as follows:
        #
        #               ------------
        #           === |  select  | ===
        #          ||   ------------   ||
        #          \/                  \/
        #     ------------       -------------
        #     |   true   |       |   false   |
        #     ------------       -------------
        #          ||                  ||
        #          ||   ------------   ||
        #           ==> |   head   | <==
        #               ------------
        #
        select_state = sdfg.add_state_before(state, state.label + "_select")
        sdfg.remove_edge(sdfg.out_edges(select_state)[0])

        # expect true branch as second argument
        true_state = sdfg.add_state(state.label + "_true_branch")
        sdfg.add_edge(select_state, true_state, dace.InterstateEdge(condition=cond))
        sdfg.add_edge(true_state, state, dace.InterstateEdge())
        self.true_br_builder = dataflow_builder.visit(true_expr, sdfg=sdfg, head_state=true_state)

        # and false branch as third argument
        false_state = sdfg.add_state(state.label + "_false_branch")
        sdfg.add_edge(select_state, false_state, dace.InterstateEdge(condition=(f"not {cond}")))
        sdfg.add_edge(false_state, state, dace.InterstateEdge())
        self.false_br_builder = dataflow_builder.visit(
            false_expr, sdfg=sdfg, head_state=false_state
        )

    def build(self) -> list[SDFGField]:
        # retrieve true/false states as predecessors of head state
        branch_states = tuple(edge.src for edge in self.sdfg.in_edges(self.head_state))
        assert len(branch_states) == 2
        if branch_states[0].label.endswith("_true_branch"):
            true_state, false_state = branch_states
        else:
            false_state, true_state = branch_states

        true_br_args = self.true_br_builder()
        false_br_args = self.false_br_builder()

        output_nodes = []
        for true_br, false_br in zip(true_br_args, false_br_args, strict=True):
            true_br_node, true_br_type = true_br
            assert isinstance(true_br_node, dace.nodes.AccessNode)
            false_br_node, false_br_type = false_br
            assert isinstance(false_br_node, dace.nodes.AccessNode)
            assert true_br_type == false_br_type
            array_type = self.sdfg.arrays[true_br_node.data]
            access_node = self.add_local_storage(true_br_type, array_type.shape)
            output_nodes.append((access_node, true_br_type))

            data_name = access_node.data
            true_br_output_node = true_state.add_access(data_name)
            true_state.add_nedge(
                true_br_node,
                true_br_output_node,
                dace.Memlet.from_array(
                    true_br_output_node.data, true_br_output_node.desc(self.sdfg)
                ),
            )

            false_br_output_node = false_state.add_access(data_name)
            false_state.add_nedge(
                false_br_node,
                false_br_output_node,
                dace.Memlet.from_array(
                    false_br_output_node.data, false_br_output_node.desc(self.sdfg)
                ),
            )
        return output_nodes


class SymbolRef(PrimitiveTranslator):
    """Generates the dataflow subgraph for a `ir.SymRef` node."""

    sym_name: str
    sym_type: ts.FieldType | ts.ScalarType

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sym_name: str,
        sym_type: ts.FieldType | ts.ScalarType,
    ):
        super().__init__(sdfg, state)
        self.sym_name = sym_name
        self.sym_type = sym_type

    def build(self) -> list[SDFGField]:
        if isinstance(self.sym_type, ts.FieldType):
            # add access node to current state
            sym_node = self.head_state.add_access(self.sym_name)

        else:
            # scalar symbols are passed to the SDFG as symbols: build tasklet node
            # to write the symbol to a scalar access node
            assert self.sym_name in self.sdfg.symbols
            tasklet_node = self.head_state.add_tasklet(
                f"get_{self.sym_name}",
                {},
                {"__out"},
                f"__out = {self.sym_name}",
            )
            sym_node = self.add_local_storage(self.sym_type, shape=[])
            self.head_state.add_edge(
                tasklet_node,
                "__out",
                sym_node,
                None,
                dace.Memlet(data=sym_node.data, subset="0"),
            )

        return [(sym_node, self.sym_type)]
