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


from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Protocol, TypeAlias

import dace
import dace.subsets as sbs

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_python_codegen,
    gtir_to_tasklet,
    utility as dace_fieldview_util,
)
from gt4py.next.type_system import type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace_fieldview import gtir_to_sdfg


IteratorIndexDType: TypeAlias = dace.int32  # type of iterator indexes
TemporaryData: TypeAlias = tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]


class PrimitiveTranslator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        node: gtir.Node,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sdfg_builder: gtir_to_sdfg.SDFGBuilder,
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
    ) -> list[TemporaryData]:
        """Creates the dataflow subgraph representing a GTIR primitive function.

        This method is used by derived classes to build a specialized subgraph
        for a specific GTIR primitive function.

        Arguments:
            node: The GTIR node describing the primitive to be lowered
            sdfg: The SDFG where the primitive subgraph should be instantiated
            state: The SDFG state where the result of the primitive function should be made available
            sdfg_builder: The object responsible for visiting child nodes of the primitive node.
            reduce_identity: The value of the reduction identity, in case the primitive node
                is visited in the context of a reduction expression. This value is used
                by the `neighbors` primitive to provide the value of skip neighbors.

        Returns:
            A list of data access nodes and the associated GT4Py data type, which provide
            access to the result of the primitive subgraph. The GT4Py data type is useful
            in the case the returned data is an array, because the type provdes the domain
            information (e.g. order of dimensions, dimension types).
        """


def _parse_arg_expr(
    node: gtir.Expr,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    domain: list[
        tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
    ],
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> gtir_to_tasklet.IteratorExpr | gtir_to_tasklet.MemletExpr:
    fields: list[TemporaryData] = sdfg_builder.visit(
        node, sdfg=sdfg, head_state=state, reduce_identity=reduce_identity
    )

    assert len(fields) == 1
    data_node, arg_type = fields[0]
    # require all argument nodes to be data access nodes (no symbols)
    assert isinstance(data_node, dace.nodes.AccessNode)

    if isinstance(arg_type, ts.ScalarType):
        return gtir_to_tasklet.MemletExpr(data_node, sbs.Indices([0]))
    else:
        assert isinstance(arg_type, ts.FieldType)
        indices: dict[gtx_common.Dimension, gtir_to_tasklet.IteratorIndexExpr] = {
            dim: gtir_to_tasklet.SymbolExpr(
                dace_fieldview_util.get_map_variable(dim),
                IteratorIndexDType,
            )
            for dim, _, _ in domain
        }
        return gtir_to_tasklet.IteratorExpr(
            data_node,
            arg_type.dims,
            indices,
        )


def _create_temporary_field(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: list[
        tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
    ],
    node_type: ts.ScalarType,
    output_desc: dace.data.Data,
    output_field_type: ts.DataType,
) -> tuple[dace.nodes.AccessNode, ts.FieldType]:
    domain_dims, domain_lbs, domain_ubs = zip(*domain)
    field_dims = list(domain_dims)
    field_shape = [
        # diff between upper and lower bound
        (ub - lb)
        for lb, ub in zip(domain_lbs, domain_ubs)
    ]
    field_offset: Optional[list[dace.symbolic.SymbolicType]] = None
    if any(domain_lbs):
        field_offset = [-lb for lb in domain_lbs]

    if isinstance(output_desc, dace.data.Array):
        # extend the result arrays with the local dimensions added by the field operator e.g. `neighbors`)
        assert isinstance(output_field_type, ts.FieldType)
        # TODO: enable `assert output_field_type.dtype == node_type`, remove variable `dtype`
        node_type = output_field_type.dtype
        field_dims.extend(output_field_type.dims)
        field_shape.extend(output_desc.shape)
    else:
        assert isinstance(output_desc, dace.data.Scalar)
        assert isinstance(output_field_type, ts.ScalarType)
        # TODO: enable `assert output_field_type == node_type`, remove variable `dtype`
        node_type = output_field_type

    # allocate local temporary storage for the result field
    temp_name, _ = sdfg.add_temp_transient(
        field_shape, dace_fieldview_util.as_dace_type(node_type), offset=field_offset
    )
    field_node = state.add_access(temp_name)
    field_type = ts.FieldType(field_dims, node_type)

    return field_node, field_type


def translate_as_field_op(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> list[TemporaryData]:
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")

    fun_node = node.fun
    assert len(fun_node.args) == 2
    stencil_expr, domain_expr = fun_node.args
    # expect stencil (represented as a lambda function) as first argument
    assert isinstance(stencil_expr, gtir.Lambda)
    # the domain of the field operator is passed as second argument
    assert isinstance(domain_expr, gtir.FunCall)

    # add local storage to compute the field operator over the given domain
    # TODO: use type inference to determine the result type
    node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    domain = dace_fieldview_util.get_domain(domain_expr)

    if cpm.is_applied_reduce(stencil_expr.expr):
        if reduce_identity:
            raise NotImplementedError("nested reductions not supported.")
        _, _, reduce_identity = gtir_to_tasklet.get_reduce_params(stencil_expr.expr)

    # first visit the list of arguments and build a symbol map
    stencil_args = [
        _parse_arg_expr(arg, sdfg, state, sdfg_builder, domain, reduce_identity)
        for arg in node.args
    ]

    # represent the field operator as a mapped tasklet graph, which will range over the field domain
    taskgen = gtir_to_tasklet.LambdaToTasklet(sdfg, state, sdfg_builder, reduce_identity)
    input_connections, output_expr = taskgen.visit(stencil_expr, args=stencil_args)
    assert isinstance(output_expr, gtir_to_tasklet.ValueExpr)
    output_desc = output_expr.node.desc(sdfg)

    # retrieve the tasklet node which writes the result
    last_node = state.in_edges(output_expr.node)[0].src
    if isinstance(last_node, dace.nodes.Tasklet):
        # the last transient node can be deleted
        last_node_connector = state.in_edges(output_expr.node)[0].src_conn
        state.remove_node(output_expr.node)
    else:
        last_node = output_expr.node
        last_node_connector = None

    # allocate local temporary storage for the result field
    field_node, field_type = _create_temporary_field(
        sdfg, state, domain, node_type, output_desc, output_expr.field_type
    )

    # assume tasklet with single output
    output_subset = [dace_fieldview_util.get_map_variable(dim) for dim, _, _ in domain]
    if isinstance(output_desc, dace.data.Array):
        # additional local dimension for neighbors
        assert set(output_desc.offset) == {0}
        output_subset.extend(f"0:{size}" for size in output_desc.shape)

    # create map range corresponding to the field operator domain
    map_ranges = {dace_fieldview_util.get_map_variable(dim): f"{lb}:{ub}" for dim, lb, ub in domain}
    me, mx = sdfg_builder.add_map("field_op", state, map_ranges)

    if len(input_connections) == 0:
        # dace requires an empty edge from map entry node to tasklet node, in case there no input memlets
        state.add_nedge(me, last_node, dace.Memlet())
    else:
        for data_node, data_subset, lambda_node, lambda_connector in input_connections:
            memlet = dace.Memlet(data=data_node.data, subset=data_subset)
            state.add_memlet_path(
                data_node,
                me,
                lambda_node,
                dst_conn=lambda_connector,
                memlet=memlet,
            )
    state.add_memlet_path(
        last_node,
        mx,
        field_node,
        src_conn=last_node_connector,
        memlet=dace.Memlet(data=field_node.data, subset=",".join(output_subset)),
    )

    return [(field_node, field_type)]


def translate_cond(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> list[TemporaryData]:
    """Generates the dataflow subgraph for the `cond` builtin function."""
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "cond")
    assert len(node.args) == 0

    fun_node = node.fun
    assert len(fun_node.args) == 3
    cond_expr, true_expr, false_expr = fun_node.args

    # expect condition as first argument
    cond = gtir_python_codegen.get_source(cond_expr)

    # use current head state to terminate the dataflow, and add a entry state
    # to connect the true/false branch states as follows:
    #
    #               ------------
    #           === |  cond  | ===
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
    cond_state = sdfg.add_state_before(state, state.label + "_cond")
    sdfg.remove_edge(sdfg.out_edges(cond_state)[0])

    # expect true branch as second argument
    true_state = sdfg.add_state(state.label + "_true_branch")
    sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(condition=f"bool({cond})"))
    sdfg.add_edge(true_state, state, dace.InterstateEdge())

    # and false branch as third argument
    false_state = sdfg.add_state(state.label + "_false_branch")
    sdfg.add_edge(cond_state, false_state, dace.InterstateEdge(condition=(f"not bool({cond})")))
    sdfg.add_edge(false_state, state, dace.InterstateEdge())

    true_br_args = sdfg_builder.visit(
        true_expr, sdfg=sdfg, head_state=true_state, reduce_identity=reduce_identity
    )
    false_br_args = sdfg_builder.visit(
        false_expr, sdfg=sdfg, head_state=false_state, reduce_identity=reduce_identity
    )

    output_nodes = []
    for true_br, false_br in zip(true_br_args, false_br_args, strict=True):
        true_br_node, true_br_type = true_br
        assert isinstance(true_br_node, dace.nodes.AccessNode)
        false_br_node, _ = false_br
        assert isinstance(false_br_node, dace.nodes.AccessNode)
        desc = true_br_node.desc(sdfg)
        assert false_br_node.desc(sdfg) == desc
        data_name, _ = sdfg.add_temp_transient_like(desc)
        output_nodes.append((state.add_access(data_name), true_br_type))

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


def translate_symbol_ref(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr] = None,
) -> list[TemporaryData]:
    """Generates the dataflow subgraph for a `ir.SymRef` node."""
    assert isinstance(node, (gtir.Literal, gtir.SymRef))

    data_type: ts.FieldType | ts.ScalarType
    if isinstance(node, gtir.Literal):
        sym_value = node.value
        data_type = node.type
        tasklet_name = "get_literal"
    else:
        sym_value = str(node.id)
        data_type = sdfg_builder.get_symbol_type(sym_value)
        tasklet_name = f"get_{sym_value}"

    if isinstance(data_type, ts.FieldType):
        # add access node to current state
        sym_node = state.add_access(sym_value)

    else:
        # scalar symbols are passed to the SDFG as symbols: build tasklet node
        # to write the symbol to a scalar access node
        tasklet_node = sdfg_builder.add_tasklet(
            tasklet_name,
            state,
            {},
            {"__out"},
            f"__out = {sym_value}",
        )
        temp_name, _ = sdfg.add_temp_transient((1,), dace_fieldview_util.as_dace_type(data_type))
        sym_node = state.add_access(temp_name)
        state.add_edge(
            tasklet_node,
            "__out",
            sym_node,
            None,
            dace.Memlet(data=sym_node.data, subset="0"),
        )

    return [(sym_node, data_type)]


if TYPE_CHECKING:
    # Use type-checking to assert that all translator functions implement the `PrimitiveTranslator` protocol
    __primitive_translators: list[PrimitiveTranslator] = [
        translate_as_field_op,
        translate_cond,
        translate_symbol_ref,
    ]
