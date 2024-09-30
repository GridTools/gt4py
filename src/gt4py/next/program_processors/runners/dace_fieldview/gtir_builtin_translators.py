# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Iterable, Optional, Protocol, TypeAlias

import dace
import dace.subsets as sbs

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import type_specifications as gtir_ts
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_python_codegen,
    gtir_to_tasklet,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace_fieldview import gtir_to_sdfg


IteratorIndexDType: TypeAlias = dace.int32  # type of iterator indexes


@dataclasses.dataclass(frozen=True)
class Field:
    data_node: dace.nodes.AccessNode
    data_type: ts.FieldType | ts.ScalarType


FieldopResult: TypeAlias = Field | tuple[Field | tuple, ...]


class PrimitiveTranslator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        node: gtir.Node,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sdfg_builder: gtir_to_sdfg.SDFGBuilder,
        reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
    ) -> FieldopResult:
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
                by the `neighbors` primitive to provide the default value of skip neighbors.

        Returns:
            A list of data access nodes and the associated GT4Py data type, which provide
            access to the result of the primitive subgraph. The GT4Py data type is useful
            in the case the returned data is an array, because the type provdes the domain
            information (e.g. order of dimensions, dimension types).
        """


def _parse_fieldop_arg(
    node: gtir.Expr,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    domain: list[
        tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
    ],
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> gtir_to_tasklet.IteratorExpr | gtir_to_tasklet.MemletExpr:
    arg = sdfg_builder.visit(
        node,
        sdfg=sdfg,
        head_state=state,
        reduce_identity=reduce_identity,
    )

    # arguments passed to field operator should be plain fields, not tuples of fields
    if not isinstance(arg, Field):
        raise ValueError(f"Received {node} as argument to field operator, expected a field.")

    if isinstance(arg.data_type, ts.ScalarType):
        return gtir_to_tasklet.MemletExpr(arg.data_node, sbs.Indices([0]))
    elif isinstance(arg.data_type, ts.FieldType):
        indices: dict[gtx_common.Dimension, gtir_to_tasklet.IteratorIndexExpr] = {
            dim: gtir_to_tasklet.SymbolExpr(
                dace_gtir_utils.get_map_variable(dim),
                IteratorIndexDType,
            )
            for dim, _, _ in domain
        }
        dims = arg.data_type.dims + (
            # we add an extra anonymous dimension in the iterator definition to enable
            # dereferencing elements in `ListType`
            [gtx_common.Dimension("")] if isinstance(arg.data_type.dtype, gtir_ts.ListType) else []
        )
        return gtir_to_tasklet.IteratorExpr(arg.data_node, dims, indices)
    else:
        raise NotImplementedError(f"Node type {type(arg.data_type)} not supported.")


def _create_temporary_field(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: list[
        tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
    ],
    node_type: ts.FieldType,
    output_desc: dace.data.Data,
) -> Field:
    domain_dims, _, domain_ubs = zip(*domain)
    field_dims = list(domain_dims)
    # It should be enough to allocate an array with shape (upper_bound - lower_bound)
    # but this would require to use array offset for compensate for the start index.
    # Suppose that a field operator executes on domain [2,N-2], the dace array to store
    # the result only needs size (N-4), but this would require to compensate all array
    # accesses with offset -2 (which corresponds to -lower_bound). Instead, we choose
    # to allocate (N-2), leaving positions [0:2] unused. The reason is that array offset
    # is known to cause issues to SDFG inlining. Besides, map fusion will in any case
    # eliminate most of transient arrays.
    field_shape = list(domain_ubs)

    if isinstance(output_desc, dace.data.Array):
        assert isinstance(node_type.dtype, gtir_ts.ListType)
        assert isinstance(node_type.dtype.element_type, ts.ScalarType)
        field_dtype = node_type.dtype.element_type
        # extend the array with the local dimensions added by the field operator (e.g. `neighbors`)
        field_shape.extend(output_desc.shape)
    elif isinstance(output_desc, dace.data.Scalar):
        field_dtype = node_type.dtype
    else:
        raise ValueError(f"Cannot create field for dace type {output_desc}.")

    # allocate local temporary storage
    temp_name, _ = sdfg.add_temp_transient(field_shape, dace_utils.as_dace_type(field_dtype))
    field_node = state.add_access(temp_name)
    field_type = ts.FieldType(field_dims, node_type.dtype)

    return Field(field_node, field_type)


def translate_as_field_op(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    """Generates the dataflow subgraph for the `as_fieldop` builtin function."""
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, ts.FieldType)

    fun_node = node.fun
    assert len(fun_node.args) == 2
    stencil_expr, domain_expr = fun_node.args
    # expect stencil (represented as a lambda function) as first argument
    assert isinstance(stencil_expr, gtir.Lambda)
    # the domain of the field operator is passed as second argument
    assert isinstance(domain_expr, gtir.FunCall)

    # add local storage to compute the field operator over the given domain
    domain = dace_gtir_utils.get_domain(domain_expr)
    assert isinstance(node.type, ts.FieldType)

    if cpm.is_applied_reduce(stencil_expr.expr):
        if reduce_identity is not None:
            raise NotImplementedError("nested reductions not supported.")

        # the reduce identity value is used to fill the skip values in neighbors list
        _, _, reduce_identity = gtir_to_tasklet.get_reduce_params(stencil_expr.expr)

    # first visit the list of arguments and build a symbol map
    stencil_args = [
        _parse_fieldop_arg(arg, sdfg, state, sdfg_builder, domain, reduce_identity)
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
    result_field = _create_temporary_field(sdfg, state, domain, node.type, output_desc)

    # assume tasklet with single output
    output_subset = [dace_gtir_utils.get_map_variable(dim) for dim, _, _ in domain]
    if isinstance(output_desc, dace.data.Array):
        # additional local dimension for neighbors
        assert set(output_desc.offset) == {0}
        output_subset.extend(f"0:{size}" for size in output_desc.shape)

    # create map range corresponding to the field operator domain
    map_ranges = {dace_gtir_utils.get_map_variable(dim): f"{lb}:{ub}" for dim, lb, ub in domain}
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
        result_field.data_node,
        src_conn=last_node_connector,
        memlet=dace.Memlet(data=result_field.data_node.data, subset=",".join(output_subset)),
    )

    return result_field


def translate_if(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    """Generates the dataflow subgraph for the `if_` builtin function."""
    assert cpm.is_call_to(node, "if_")
    assert len(node.args) == 3
    cond_expr, true_expr, false_expr = node.args

    # expect condition as first argument
    if_stmt = gtir_python_codegen.get_source(cond_expr)

    # use current head state to terminate the dataflow, and add a entry state
    # to connect the true/false branch states as follows:
    #
    #               ------------
    #           === |   cond   | ===
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
    sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(condition=f"bool({if_stmt})"))
    sdfg.add_edge(true_state, state, dace.InterstateEdge())

    # and false branch as third argument
    false_state = sdfg.add_state(state.label + "_false_branch")
    sdfg.add_edge(cond_state, false_state, dace.InterstateEdge(condition=(f"not bool({if_stmt})")))
    sdfg.add_edge(false_state, state, dace.InterstateEdge())

    true_br_args = sdfg_builder.visit(
        true_expr,
        sdfg=sdfg,
        head_state=true_state,
        reduce_identity=reduce_identity,
    )
    false_br_args = sdfg_builder.visit(
        false_expr,
        sdfg=sdfg,
        head_state=false_state,
        reduce_identity=reduce_identity,
    )

    def make_temps(x: Field) -> Field:
        desc = x.data_node.desc(sdfg)
        data_name, _ = sdfg.add_temp_transient_like(desc)
        data_node = state.add_access(data_name)

        return Field(data_node, x.data_type)

    result_temps = gtx_utils.tree_map(make_temps)(true_br_args)

    fields: Iterable[tuple[Field, Field, Field]] = zip(
        gtx_utils.flatten_nested_tuple((true_br_args,)),
        gtx_utils.flatten_nested_tuple((false_br_args,)),
        gtx_utils.flatten_nested_tuple((result_temps,)),
        strict=True,
    )

    for true_br, false_br, temp in fields:
        assert true_br.data_type == false_br.data_type
        true_br_node = true_br.data_node
        false_br_node = false_br.data_node

        temp_name = temp.data_node.data
        temp_desc = temp.data_node.desc(sdfg)
        true_br_output_node = true_state.add_access(temp_name)
        true_state.add_nedge(
            true_br_node,
            true_br_output_node,
            dace.Memlet.from_array(temp_name, temp_desc),
        )

        false_br_output_node = false_state.add_access(temp_name)
        false_state.add_nedge(
            false_br_node,
            false_br_output_node,
            dace.Memlet.from_array(temp_name, temp_desc),
        )

    return result_temps


def _get_data_nodes(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    sym_name: str,
    sym_type: ts.DataType,
) -> FieldopResult:
    if isinstance(sym_type, ts.FieldType):
        sym_node = state.add_access(sym_name)
        return Field(sym_node, sym_type)
    elif isinstance(sym_type, ts.ScalarType):
        if sym_name in sdfg.arrays:
            # access the existing scalar container
            sym_node = state.add_access(sym_name)
        else:
            sym_node = _get_symbolic_value(
                sdfg, state, sdfg_builder, sym_name, sym_type, temp_name=f"__{sym_name}"
            )
        return Field(sym_node, sym_type)
    elif isinstance(sym_type, ts.TupleType):
        tuple_fields = dace_gtir_utils.get_tuple_fields(sym_name, sym_type)
        return tuple(
            _get_data_nodes(sdfg, state, sdfg_builder, fname, ftype)
            for fname, ftype in tuple_fields
        )
    else:
        raise NotImplementedError(f"Symbol type {type(sym_type)} not supported.")


def _get_symbolic_value(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    symbolic_expr: dace.symbolic.SymExpr,
    scalar_type: ts.ScalarType,
    temp_name: Optional[str] = None,
) -> dace.nodes.AccessNode:
    tasklet_node = sdfg_builder.add_tasklet(
        "get_value",
        state,
        {},
        {"__out"},
        f"__out = {symbolic_expr}",
    )
    temp_name, _ = sdfg.add_scalar(
        temp_name or sdfg.temp_data_name(),
        dace_utils.as_dace_type(scalar_type),
        find_new_name=True,
        transient=True,
    )
    data_node = state.add_access(temp_name)
    state.add_edge(
        tasklet_node,
        "__out",
        data_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )
    return data_node


def translate_literal(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    """Generates the dataflow subgraph for a `ir.Literal` node."""
    assert isinstance(node, gtir.Literal)

    data_type = node.type
    data_node = _get_symbolic_value(sdfg, state, sdfg_builder, node.value, data_type)

    return Field(data_node, data_type)


def translate_make_tuple(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    assert cpm.is_call_to(node, "make_tuple")
    return tuple(
        sdfg_builder.visit(
            arg,
            sdfg=sdfg,
            head_state=state,
            reduce_identity=reduce_identity,
        )
        for arg in node.args
    )


def translate_tuple_get(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    assert cpm.is_call_to(node, "tuple_get")
    assert len(node.args) == 2

    if not isinstance(node.args[0], gtir.Literal):
        raise ValueError("Tuple can only be subscripted with compile-time constants.")
    assert node.args[0].type == dace_utils.as_scalar_type(gtir.INTEGER_INDEX_BUILTIN)
    index = int(node.args[0].value)

    data_nodes = sdfg_builder.visit(
        node.args[1],
        sdfg=sdfg,
        head_state=state,
        reduce_identity=reduce_identity,
    )
    if isinstance(data_nodes, Field):
        raise ValueError(f"Invalid tuple expression {node}")
    unused_arg_nodes: list[Field] = list(
        gtx_utils.flatten_nested_tuple(tuple(arg for i, arg in enumerate(data_nodes) if i != index))
    )
    state.remove_nodes_from(
        [arg.data_node for arg in unused_arg_nodes if state.degree(arg.data_node) == 0]
    )
    return data_nodes[index]


def translate_scalar_expr(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    assert isinstance(node, gtir.FunCall)
    assert isinstance(node.type, ts.ScalarType)

    args = []
    connectors = []
    scalar_expr_args = []

    for arg_expr in node.args:
        visit_expr = True
        if isinstance(arg_expr, gtir.SymRef):
            try:
                # `gt_symbol` refers to symbols defined in the GT4Py program
                gt_symbol_type = sdfg_builder.get_symbol_type(arg_expr.id)
                if not isinstance(gt_symbol_type, ts.ScalarType):
                    raise ValueError(f"Invalid argument to scalar expression {arg_expr}.")
            except KeyError:
                # this is the case of non-variable argument, e.g. target type such as `float64`,
                # used in a casting expression like `cast_(variable, float64)`
                visit_expr = False

        if visit_expr:
            # we visit the argument expression and obtain the access node to
            # a scalar data container, which will be connected to the tasklet
            arg = sdfg_builder.visit(
                arg_expr,
                sdfg=sdfg,
                head_state=state,
                reduce_identity=reduce_identity,
            )
            if not (isinstance(arg, Field) and isinstance(arg.data_type, ts.ScalarType)):
                raise ValueError(f"Invalid argument to scalar expression {arg_expr}.")
            param = f"__in_{arg.data_node.data}"
            args.append(arg.data_node)
            connectors.append(param)
            scalar_expr_args.append(gtir.SymRef(id=param))
        else:
            assert isinstance(arg_expr, gtir.SymRef)
            scalar_expr_args.append(arg_expr)

    # we visit the scalar expression replacing the input arguments with the corresponding data connectors
    scalar_node = gtir.FunCall(fun=node.fun, args=scalar_expr_args)
    python_code = gtir_python_codegen.get_source(scalar_node)
    tasklet_node = sdfg_builder.add_tasklet(
        name="scalar_expr",
        state=state,
        inputs=set(connectors),
        outputs={"__out"},
        code=f"__out = {python_code}",
    )
    # create edges for the input data connectors
    for arg_node, conn in zip(args, connectors, strict=True):
        state.add_edge(
            arg_node,
            None,
            tasklet_node,
            conn,
            dace.Memlet(data=arg_node.data, subset="0"),
        )
    # finally, create temporary for the result value
    temp_name, _ = sdfg.add_scalar(
        sdfg.temp_data_name(),
        dace_utils.as_dace_type(node.type),
        find_new_name=True,
        transient=True,
    )
    temp_node = state.add_access(temp_name)
    state.add_edge(
        tasklet_node,
        "__out",
        temp_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )

    return Field(temp_node, node.type)


def translate_symbol_ref(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    reduce_identity: Optional[gtir_to_tasklet.SymbolExpr],
) -> FieldopResult:
    """Generates the dataflow subgraph for a `ir.SymRef` node."""
    assert isinstance(node, gtir.SymRef)

    symbol_name = str(node.id)
    # we retrieve the type of the symbol in the GT4Py prgram
    gt_symbol_type = sdfg_builder.get_symbol_type(symbol_name)

    # Create new access node in current state. It is possible that multiple
    # access nodes are created in one state for the same data container.
    # We rely on the dace simplify pass to remove duplicated access nodes.
    return _get_data_nodes(sdfg, state, sdfg_builder, symbol_name, gt_symbol_type)


if TYPE_CHECKING:
    # Use type-checking to assert that all translator functions implement the `PrimitiveTranslator` protocol
    __primitive_translators: list[PrimitiveTranslator] = [
        translate_as_field_op,
        translate_if,
        translate_literal,
        translate_make_tuple,
        translate_tuple_get,
        translate_scalar_expr,
        translate_symbol_ref,
    ]
