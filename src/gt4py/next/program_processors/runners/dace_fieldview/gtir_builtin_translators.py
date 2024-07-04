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
from typing import Callable, Optional, TypeAlias, final

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
ITERATOR_INDEX_DTYPE = dace.int32  # type of iterator indexes


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
        self,
        data_type: ts.FieldType | ts.ScalarType,
        shape: Optional[list[dace.symbolic.SymbolicType]] = None,
        offset: Optional[list[dace.symbolic.SymbolicType]] = None,
    ) -> dace.nodes.AccessNode:
        """
        Allocates temporary storage to be used in the local scope for intermediate results.

        The storage is allocate with unique names to enable SSA optimization in the compilation phase.
        """
        if isinstance(data_type, ts.FieldType):
            assert shape
            assert len(data_type.dims) == len(shape)
            dtype = dace_fieldview_util.as_dace_type(data_type.dtype)
            name, _ = self.sdfg.add_array(
                "var", shape, dtype, offset=offset, find_new_name=True, transient=True
            )
        else:
            assert not shape
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
    field_domain: list[tuple[Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]
    field_dtype: ts.ScalarType
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

        # add local storage to compute the field operator over the given domain
        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        self.field_domain = dace_fieldview_util.get_field_domain(domain_expr)
        self.field_dtype = node_type
        self.stencil_expr = stencil_expr
        self.stencil_args = stencil_args

    def build(self) -> list[SDFGField]:
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
                indices: dict[Dimension, gtir_to_tasklet.IteratorIndexExpr] = {
                    dim: gtir_to_tasklet.SymbolExpr(
                        dace.symbolic.SymExpr(DIMENSION_INDEX_FMT.format(dim=dim.value)),
                        ITERATOR_INDEX_DTYPE,
                    )
                    for dim, _, _ in self.field_domain
                }
                iterator_arg = gtir_to_tasklet.IteratorExpr(
                    data_node,
                    arg_type.dims,
                    indices,
                )
                stencil_args.append(iterator_arg)

        # create map range corresponding to the field operator domain
        map_ranges = {
            DIMENSION_INDEX_FMT.format(dim=dim.value): f"{lb}:{ub}"
            for dim, lb, ub in self.field_domain
        }
        me, mx = self.head_state.add_map("field_op", map_ranges)

        # represent the field operator as a mapped tasklet graph, which will range over the field domain
        taskgen = gtir_to_tasklet.LambdaToTasklet(
            self.sdfg, self.head_state, me, self.offset_provider
        )
        output_expr = taskgen.visit(self.stencil_expr, args=stencil_args)
        assert isinstance(output_expr, gtir_to_tasklet.ValueExpr)
        output_desc = output_expr.node.desc(self.sdfg)

        # retrieve the tasklet node which writes the result
        last_node = self.head_state.in_edges(output_expr.node)[0].src
        if isinstance(last_node, dace.nodes.Tasklet):
            # the last transient node can be deleted
            last_node_connector = self.head_state.in_edges(output_expr.node)[0].src_conn
            self.head_state.remove_node(output_expr.node)
            if len(last_node.in_connectors) == 0:
                # dace requires an empty edge from map entry node to tasklet node, in case there no input memlets
                self.head_state.add_nedge(me, last_node, dace.Memlet())
        else:
            last_node = output_expr.node
            last_node_connector = None

        # allocate local temporary storage for the result field
        field_dims = [dim for dim, _, _ in self.field_domain]
        field_shape = [
            # diff between upper and lower bound
            (ub - lb)
            for _, lb, ub in self.field_domain
        ]
        field_offset: Optional[list[dace.symbolic.SymbolicType]] = None
        if any(lb != 0 for _, lb, _ in self.field_domain):
            field_offset = [lb for _, lb, _ in self.field_domain]
        if isinstance(output_desc, dace.data.Array):
            raise NotImplementedError
        else:
            assert isinstance(output_expr.field_type, ts.ScalarType)
            # TODO: enable `assert output_expr.field_type == self.field_dtype`, remove variable `dtype`
            dtype = output_expr.field_type

        # TODO: use `self.field_dtype` directly, without passing through `dtype`
        field_type = ts.FieldType(field_dims, dtype)
        field_node = self.add_local_storage(field_type, field_shape, field_offset)

        # assume tasklet with single output
        output_subset = [
            DIMENSION_INDEX_FMT.format(dim=dim.value) for dim, _, _ in self.field_domain
        ]
        if isinstance(output_desc, dace.data.Array):
            # additional local dimension for neighbors
            assert output_desc.offset is None
            output_subset.extend(f"0:{size}" for size in output_desc.shape)

        self.head_state.add_memlet_path(
            last_node,
            mx,
            field_node,
            src_conn=last_node_connector,
            memlet=dace.Memlet(data=field_node.data, subset=",".join(output_subset)),
        )

        return [(field_node, field_type)]


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
        sdfg.add_edge(select_state, true_state, dace.InterstateEdge(condition=f"bool({cond})"))
        sdfg.add_edge(true_state, state, dace.InterstateEdge())
        self.true_br_builder = dataflow_builder.visit(true_expr, sdfg=sdfg, head_state=true_state)

        # and false branch as third argument
        false_state = sdfg.add_state(state.label + "_false_branch")
        sdfg.add_edge(
            select_state, false_state, dace.InterstateEdge(condition=(f"not bool({cond})"))
        )
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
            false_br_node, _ = false_br
            assert isinstance(false_br_node, dace.nodes.AccessNode)
            desc = true_br_node.desc(self.sdfg)
            assert false_br_node.desc(self.sdfg) == desc
            data_name, _ = self.sdfg.add_temp_transient_like(desc)
            output_nodes.append((self.head_state.add_access(data_name), true_br_type))

            true_br_output_node = true_state.add_access(data_name)
            true_state.add_nedge(
                true_br_node,
                true_br_output_node,
                dace.Memlet.from_array(data_name, desc),
            )

            false_br_output_node = false_state.add_access(data_name)
            false_state.add_nedge(
                false_br_node,
                false_br_output_node,
                dace.Memlet.from_array(data_name, desc),
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
            sym_node = self.add_local_storage(self.sym_type)
            self.head_state.add_edge(
                tasklet_node,
                "__out",
                sym_node,
                None,
                dace.Memlet(data=sym_node.data, subset="0"),
            )

        return [(sym_node, self.sym_type)]
