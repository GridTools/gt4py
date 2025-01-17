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
from typing import TYPE_CHECKING, Any, Final, Iterable, Optional, Protocol, Sequence, TypeAlias

import dace
from dace import subsets as dace_subsets

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.ffront import fbuiltins as gtx_fbuiltins
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_dataflow,
    gtir_python_codegen,
    gtir_sdfg,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace_fieldview import gtir_sdfg


def _get_domain_indices(
    dims: Sequence[gtx_common.Dimension], offsets: Optional[Sequence[dace.symbolic.SymExpr]] = None
) -> dace_subsets.Indices:
    """
    Helper function to construct the list of indices for a field domain, applying
    an optional offset in each dimension as start index.

    Args:
        dims: The field dimensions.
        offsets: The range start index in each dimension.

    Returns:
        A list of indices for field access in dace arrays. As this list is returned
        as `dace.subsets.Indices`, it should be converted to `dace.subsets.Range` before
        being used in memlet subset because ranges are better supported throughout DaCe.
    """
    index_variables = [dace.symbolic.SymExpr(dace_gtir_utils.get_map_variable(dim)) for dim in dims]
    if offsets is None:
        return dace_subsets.Indices(index_variables)
    else:
        return dace_subsets.Indices(
            [
                index - offset if offset != 0 else index
                for index, offset in zip(index_variables, offsets, strict=True)
            ]
        )


@dataclasses.dataclass(frozen=True)
class FieldopData:
    """
    Abstraction to represent data (scalars, arrays) during the lowering to SDFG.

    Attribute 'local_offset' must always be set for `FieldType` data with a local
    dimension generated from neighbors access in unstructured domain, and indicates
    the name of the offset provider used to generate the list of neighbor values.

    Args:
        dc_node: DaCe access node to the data storage.
        gt_type: GT4Py type definition, which includes the field domain information.
        offset: List of index offsets, in each dimension, when the dimension range
            does not start from zero; assume zero offset, if not set.
    """

    dc_node: dace.nodes.AccessNode
    gt_type: ts.FieldType | ts.ScalarType
    offset: Optional[list[dace.symbolic.SymExpr]]

    def make_copy(self, data_node: dace.nodes.AccessNode) -> FieldopData:
        """Create a copy of this data descriptor with a different access node."""
        assert data_node != self.dc_node
        return FieldopData(data_node, self.gt_type, self.offset)

    def get_local_view(
        self, domain: FieldopDomain
    ) -> gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr:
        """Helper method to access a field in local view, given the compute domain of a field operator."""
        if isinstance(self.gt_type, ts.ScalarType):
            return gtir_dataflow.MemletExpr(
                dc_node=self.dc_node, gt_dtype=self.gt_type, subset=dace_subsets.Indices([0])
            )

        if isinstance(self.gt_type, ts.FieldType):
            domain_dims = [dim for dim, _, _ in domain]
            domain_indices = _get_domain_indices(domain_dims)
            it_indices: dict[gtx_common.Dimension, gtir_dataflow.DataExpr] = {
                dim: gtir_dataflow.SymbolExpr(index, INDEX_DTYPE)
                for dim, index in zip(domain_dims, domain_indices)
            }
            field_domain = [
                (dim, dace.symbolic.SymExpr(0) if self.offset is None else self.offset[i])
                for i, dim in enumerate(self.gt_type.dims)
            ]
            local_dims = [
                dim for dim in self.gt_type.dims if dim.kind == gtx_common.DimensionKind.LOCAL
            ]
            if len(local_dims) == 0:
                return gtir_dataflow.IteratorExpr(
                    self.dc_node, self.gt_type.dtype, field_domain, it_indices
                )

            elif len(local_dims) == 1:
                field_dtype = ts.ListType(
                    element_type=self.gt_type.dtype, offset_type=local_dims[0]
                )
                field_domain = [
                    (dim, offset)
                    for dim, offset in field_domain
                    if dim.kind != gtx_common.DimensionKind.LOCAL
                ]
                return gtir_dataflow.IteratorExpr(
                    self.dc_node, field_dtype, field_domain, it_indices
                )

            else:
                raise ValueError(
                    f"Unexpected data field {self.dc_node.data} with more than one local dimension."
                )

        raise NotImplementedError(f"Node type {type(self.gt_type)} not supported.")


FieldopDomain: TypeAlias = list[
    tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
]
"""
Domain of a field operator represented as a list of tuples with 3 elements:
 - dimension definition
 - symbolic expression for lower bound (inclusive)
 - symbolic expression for upper bound (exclusive)
"""


FieldopResult: TypeAlias = FieldopData | tuple[FieldopData | tuple, ...]
"""Result of a field operator, can be either a field or a tuple fields."""


INDEX_DTYPE: Final[dace.typeclass] = dace.dtype_to_typeclass(gtx_fbuiltins.IndexType)
"""Data type used for field indexing."""


def get_tuple_type(data: tuple[FieldopResult, ...]) -> ts.TupleType:
    """
    Compute the `ts.TupleType` corresponding to the tuple structure of `FieldopResult`.
    """
    return ts.TupleType(
        types=[get_tuple_type(d) if isinstance(d, tuple) else d.gt_type for d in data]
    )


def flatten_tuples(name: str, arg: FieldopResult) -> list[tuple[str, FieldopData]]:
    """
    Visit a `FieldopResult`, potentially containing nested tuples, and construct a list
    of pairs `(str, FieldopData)` containing the symbol name of each tuple field and
    the corresponding `FieldopData`.
    """
    if isinstance(arg, tuple):
        tuple_type = get_tuple_type(arg)
        tuple_symbols = dace_gtir_utils.flatten_tuple_fields(name, tuple_type)
        tuple_data_fields = gtx_utils.flatten_nested_tuple(arg)
        return [
            (str(tsym.id), tfield)
            for tsym, tfield in zip(tuple_symbols, tuple_data_fields, strict=True)
        ]
    else:
        return [(name, arg)]


class PrimitiveTranslator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        node: gtir.Node,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sdfg_builder: gtir_sdfg.SDFGBuilder,
    ) -> FieldopResult:
        """Creates the dataflow subgraph representing a GTIR primitive function.

        This method is used by derived classes to build a specialized subgraph
        for a specific GTIR primitive function.

        Args:
            node: The GTIR node describing the primitive to be lowered
            sdfg: The SDFG where the primitive subgraph should be instantiated
            state: The SDFG state where the result of the primitive function should be made available
            sdfg_builder: The object responsible for visiting child nodes of the primitive node.

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
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    domain: FieldopDomain,
    use_full_shape: bool = False,
) -> (
    gtir_dataflow.IteratorExpr
    | gtir_dataflow.MemletExpr
    | tuple[gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr | tuple[Any, ...], ...]
):
    """Helper method to visit an expression passed as argument to a field operator."""

    def _parse_fieldop_arg_impl(
        arg: FieldopData,
    ) -> gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr:
        arg_expr = arg.get_local_view(domain)
        if not use_full_shape or isinstance(arg_expr, gtir_dataflow.MemletExpr):
            return arg_expr
        # In case of scan field operator, the arguments to the vertical stencil are passed by value.
        return gtir_dataflow.MemletExpr(
            arg_expr.field, arg_expr.gt_dtype, arg_expr.get_memlet_subset(sdfg)
        )

    arg = sdfg_builder.visit(node, sdfg=sdfg, head_state=state)

    if isinstance(arg, FieldopData):
        return _parse_fieldop_arg_impl(arg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(lambda x: _parse_fieldop_arg_impl(x))(arg)


def _get_field_layout(
    domain: FieldopDomain,
) -> tuple[list[gtx_common.Dimension], list[dace.symbolic.SymExpr], list[dace.symbolic.SymExpr]]:
    """
    Parse the field operator domain and generates the shape of the result field.

    It should be enough to allocate an array with shape (upper_bound - lower_bound)
    but this would require to use array offset for compensate for the start index.
    Suppose that a field operator executes on domain [2,N-2], the dace array to store
    the result only needs size (N-4), but this would require to compensate all array
    accesses with offset -2 (which corresponds to -lower_bound). Instead, we choose
    to allocate (N-2), leaving positions [0:2] unused. The reason is that array offset
    is known to cause issues to SDFG inlining. Besides, map fusion will in any case
    eliminate most of transient arrays.

    Args:
        domain: The field operator domain.

    Returns:
        A tuple of three lists containing:
            - the domain dimensions
            - the domain offset in each dimension
            - the domain size in each dimension
    """
    domain_dims, domain_lbs, domain_ubs = zip(*domain)
    domain_sizes = [(ub - lb) for lb, ub in zip(domain_lbs, domain_ubs)]
    return list(domain_dims), list(domain_lbs), domain_sizes


def _create_field_operator_impl(
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: FieldopDomain,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
) -> FieldopData:
    """
    Helper method to allocate a temporary array that stores one field computed by a field operator.

    This method is called by `_create_field_operator()`.

    Args:
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        sdfg: The SDFG that represents the scope of the field data.
        state: The SDFG state where to create an access node to the field data.
        domain: The domain of the field operator that computes the field.
        output_edge: The dataflow write edge representing the output data.
        output_type: The GT4Py field type descriptor.
        map_exit: The `MapExit` node of the field operator map scope.

    Returns:
        The field data descriptor, which includes the field access node in the given `state`
        and the field domain offset.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(sdfg)

    domain_dims, domain_offset, domain_shape = _get_field_layout(domain)
    domain_indices = _get_domain_indices(domain_dims, domain_offset)
    domain_subset = dace_subsets.Range.from_indices(domain_indices)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        assert output_edge.result.gt_dtype == output_type.dtype
        field_dtype = output_edge.result.gt_dtype
        field_dims, field_shape, field_offset = (domain_dims, domain_shape, domain_offset)
        assert isinstance(dataflow_output_desc, dace.data.Scalar)
        field_subset = domain_subset
    else:
        assert isinstance(output_type.dtype, ts.ListType)
        assert isinstance(output_edge.result.gt_dtype.element_type, ts.ScalarType)
        assert output_edge.result.gt_dtype.element_type == output_type.dtype.element_type
        field_dtype = output_edge.result.gt_dtype.element_type
        assert isinstance(dataflow_output_desc, dace.data.Array)
        assert len(dataflow_output_desc.shape) == 1
        # extend the array with the local dimensions added by the field operator (e.g. `neighbors`)
        assert output_edge.result.gt_dtype.offset_type is not None
        field_dims = [*domain_dims, output_edge.result.gt_dtype.offset_type]
        field_shape = [*domain_shape, dataflow_output_desc.shape[0]]
        field_offset = [*domain_offset, dataflow_output_desc.offset[0]]
        field_subset = domain_subset + dace_subsets.Range.from_array(dataflow_output_desc)

    # allocate local temporary storage
    assert dataflow_output_desc.dtype == dace_utils.as_dace_type(field_dtype)
    field_name, _ = sdfg_builder.add_temp_array(sdfg, field_shape, dataflow_output_desc.dtype)
    field_node = state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    output_edge.connect(map_exit, field_node, field_subset)

    return FieldopData(
        field_node,
        ts.FieldType(field_dims, field_dtype),
        offset=(field_offset if set(field_offset) != {0} else None),
    )


def _create_scan_field_operator_impl(
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: FieldopDomain,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    output_type: ts.FieldType,
    map_exit: dace.nodes.MapExit,
    scan_dim: gtx_common.Dimension,
) -> FieldopData:
    """
    Similar to `_create_scan_field_operator_impl()` but for scan field operators.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(sdfg)

    domain_dims, domain_offset, domain_shape = _get_field_layout(domain)
    domain_indices = _get_domain_indices(domain_dims, domain_offset)
    domain_subset = dace_subsets.Range.from_indices(domain_indices)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        # the scan field operator produces a 1D vertical field
        assert isinstance(dataflow_output_desc, dace.data.Array)
        assert len(dataflow_output_desc.shape) == 1
        assert output_edge.result.gt_dtype == output_type.dtype
        field_dtype = output_edge.result.gt_dtype
        field_dims, field_shape, field_offset = (domain_dims, domain_shape, domain_offset)
        # the vertical dimension should not belong to the field operator domain
        # but we need to write it to the output field
        scan_dim_index = domain_dims.index(scan_dim)
        field_subset = (
            dace_subsets.Range(domain_subset[:scan_dim_index])
            + dace_subsets.Range.from_array(dataflow_output_desc)
            + dace_subsets.Range(domain_subset[scan_dim_index + 1 :])
        )
    else:
        raise NotImplementedError("List of values not supported in scan field operators.")

    # allocate local temporary storage
    assert dataflow_output_desc.dtype == dace_utils.as_dace_type(field_dtype)
    field_name, field_desc = sdfg_builder.add_temp_array(
        sdfg, field_shape, dataflow_output_desc.dtype
    )
    scan_output_stride = field_desc.strides[scan_dim_index]
    dataflow_output_desc.strides = (scan_output_stride,)
    field_node = state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    output_edge.connect(map_exit, field_node, field_subset)

    return FieldopData(
        field_node,
        ts.FieldType(field_dims, field_dtype),
        offset=(field_offset if set(field_offset) != {0} else None),
    )


def _create_field_operator(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output_edges: gtir_dataflow.DataflowOutputEdge
    | tuple[gtir_dataflow.DataflowOutputEdge | tuple[Any, ...], ...],
    scan_dim: Optional[gtx_common.Dimension] = None,
) -> FieldopResult:
    """
    Helper method to build the output of a field operator, which can consist of
    a single field or a tuple of fields.

    A tuple of fields is returned when one stencil computes a grid point on multiple
    fields: for each field, this method will call `_create_field_operator_impl()`.

    Args:
        sdfg: The SDFG that represents the scope of the field data.
        state: The SDFG state where to create an access node to the field data.
        domain: The domain of the field operator that computes the field.
        node_type: The GT4Py type of the IR node that produces this field.
        sdfg_builder: The object used to build the map scope in the provided SDFG.
        input_edges: List of edges to pass input data into the dataflow.
        output_edges: Single edge or tuple of edges representing the dataflow output data.
        scan_dim: Column dimension used in scan field operators.

    Returns:
        The descriptor of the field operator result, which can be either a single field
        or a tuple fields.
    """
    domain_dims, _, _ = _get_field_layout(domain)

    assert scan_dim is None or scan_dim in domain_dims
    if scan_dim and len(domain_dims) == 1:
        # We construct the scan field operator only on the horizontal domain.
        # If the field operator computes only the scan dimension,
        # there is no horizontal domain, therefore the map scope is not needed.
        map_entry, map_exit = (None, None)
    else:
        # create map range corresponding to the field operator domain
        map_entry, map_exit = sdfg_builder.add_map(
            "fieldop",
            state,
            ndrange={
                dace_gtir_utils.get_map_variable(dim): f"{lower_bound}:{upper_bound}"
                for dim, lower_bound, upper_bound in domain
                if dim != scan_dim
            },
        )

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert isinstance(output_edges, gtir_dataflow.DataflowOutputEdge)
        if scan_dim:
            return _create_scan_field_operator_impl(
                sdfg_builder, sdfg, state, domain, output_edges, node_type, map_exit, scan_dim
            )
        else:
            return _create_field_operator_impl(
                sdfg_builder, sdfg, state, domain, output_edges, node_type, map_exit
            )
    else:
        # handle tuples of fields
        output_symbol_tree = dace_gtir_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda output_edge, output_sym: (
                _create_scan_field_operator_impl(
                    sdfg_builder,
                    sdfg,
                    state,
                    domain,
                    output_edge,
                    output_sym.type,
                    map_exit,
                    scan_dim,
                )
                if scan_dim
                else _create_field_operator_impl(
                    sdfg_builder, sdfg, state, domain, output_edge, output_sym.type, map_exit
                )
            )
        )(output_edges, output_symbol_tree)


def extract_domain(node: gtir.Node) -> FieldopDomain:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive: [lower_bound, upper_bound[
    """

    domain = []

    def parse_range_boundary(expr: gtir.Expr) -> str:
        return dace.symbolic.pystr_to_symbolic(gtir_python_codegen.get_source(expr))

    if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, gtir.AxisLiteral)
            lower_bound, upper_bound = (parse_range_boundary(arg) for arg in named_range.args[1:3])
            dim = gtx_common.Dimension(axis.value, axis.kind)
            domain.append((dim, lower_bound, upper_bound))

    elif isinstance(node, domain_utils.SymbolicDomain):
        assert str(node.grid_type) in {"cartesian_domain", "unstructured_domain"}
        for dim, drange in node.ranges.items():
            domain.append(
                (dim, parse_range_boundary(drange.start), parse_range_boundary(drange.stop))
            )

    else:
        raise ValueError(f"Invalid domain {node}.")

    return domain


def translate_as_fieldop(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    """
    Generates the dataflow subgraph for the `as_fieldop` builtin function.

    Expects a `FunCall` node with two arguments:
    1. a lambda function representing the stencil, which is lowered to a dataflow subgraph
    2. the domain of the field operator, which is used as map range

    The dataflow can be as simple as a single tasklet, or implement a local computation
    as a composition of tasklets and even include a map to range on local dimensions (e.g.
    neighbors and map builtins).
    The stencil dataflow is instantiated inside a map scope, which applies the stencil
    over the field domain.
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    fieldop_expr, domain_expr = fun_node.args

    if cpm.is_call_to(fieldop_expr, "scan"):
        return translate_scan(node, sdfg, state, sdfg_builder)

    if cpm.is_ref_to(fieldop_expr, "deref"):
        # Special usage of 'deref' as argument to fieldop expression, to pass a scalar
        # value to 'as_fieldop' function. It results in broadcasting the scalar value
        # over the field domain.
        assert isinstance(node.type, ts.FieldType)
        stencil_expr = im.lambda_("a")(im.deref("a"))
        stencil_expr.expr.type = node.type.dtype
    elif isinstance(fieldop_expr, gtir.Lambda):
        # Default case, handled below: the argument expression is a lambda function
        # representing the stencil operation to be computed over the field domain.
        stencil_expr = fieldop_expr
    else:
        raise NotImplementedError(
            f"Expression type '{type(fieldop_expr)}' not supported as argument to 'as_fieldop' node."
        )

    # parse the domain of the field operator
    domain = extract_domain(domain_expr)

    # visit the list of arguments to be passed to the lambda expression
    fieldop_args = [_parse_fieldop_arg(arg, sdfg, state, sdfg_builder, domain) for arg in node.args]

    # represent the field operator as a mapped tasklet graph, which will range over the field domain
    input_edges, output_edges = gtir_dataflow.translate_lambda_to_dataflow(
        sdfg, state, sdfg_builder, stencil_expr, fieldop_args
    )

    return _create_field_operator(
        sdfg, state, domain, node.type, sdfg_builder, input_edges, output_edges
    )


def translate_if(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
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
    sdfg.add_edge(cond_state, true_state, dace.InterstateEdge(condition=f"{if_stmt}"))
    sdfg.add_edge(true_state, state, dace.InterstateEdge())

    # and false branch as third argument
    false_state = sdfg.add_state(state.label + "_false_branch")
    sdfg.add_edge(cond_state, false_state, dace.InterstateEdge(condition=(f"not ({if_stmt})")))
    sdfg.add_edge(false_state, state, dace.InterstateEdge())

    true_br_args = sdfg_builder.visit(
        true_expr,
        sdfg=sdfg,
        head_state=true_state,
    )
    false_br_args = sdfg_builder.visit(
        false_expr,
        sdfg=sdfg,
        head_state=false_state,
    )

    def construct_output(inner_data: FieldopData) -> FieldopData:
        inner_desc = inner_data.dc_node.desc(sdfg)
        outer, _ = sdfg_builder.add_temp_array_like(sdfg, inner_desc)
        outer_node = state.add_access(outer)

        return inner_data.make_copy(outer_node)

    result_temps = gtx_utils.tree_map(construct_output)(true_br_args)

    fields: Iterable[tuple[FieldopData, FieldopData, FieldopData]] = zip(
        gtx_utils.flatten_nested_tuple((true_br_args,)),
        gtx_utils.flatten_nested_tuple((false_br_args,)),
        gtx_utils.flatten_nested_tuple((result_temps,)),
        strict=True,
    )

    for true_br, false_br, temp in fields:
        if true_br.gt_type != false_br.gt_type:
            raise ValueError(
                f"Different type of result fields on if-branches '{true_br.gt_type}' vs '{false_br.gt_type}'."
            )
        true_br_node = true_br.dc_node
        false_br_node = false_br.dc_node

        temp_name = temp.dc_node.data
        true_br_output_node = true_state.add_access(temp_name)
        true_state.add_nedge(
            true_br_node,
            true_br_output_node,
            sdfg.make_array_memlet(temp_name),
        )

        false_br_output_node = false_state.add_access(temp_name)
        false_state.add_nedge(
            false_br_node,
            false_br_output_node,
            sdfg.make_array_memlet(temp_name),
        )

    return result_temps


def translate_index(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    """
    Lowers the `index` builtin function to a mapped tasklet that writes the dimension
    index values to a transient array. The extent of the index range is taken from
    the domain information that should be present in the node annex.
    """
    assert cpm.is_call_to(node, "index")
    assert isinstance(node.type, ts.FieldType)

    assert "domain" in node.annex
    domain = extract_domain(node.annex.domain)
    assert len(domain) == 1
    dim, _, _ = domain[0]
    dim_index = dace_gtir_utils.get_map_variable(dim)

    index_data, _ = sdfg_builder.add_temp_scalar(sdfg, INDEX_DTYPE)
    index_node = state.add_access(index_data)
    index_value = gtir_dataflow.ValueExpr(
        dc_node=index_node,
        gt_dtype=dace_utils.as_itir_type(INDEX_DTYPE),
    )
    index_write_tasklet = sdfg_builder.add_tasklet(
        "index",
        state,
        inputs={},
        outputs={"__val"},
        code=f"__val = {dim_index}",
    )
    state.add_edge(
        index_write_tasklet,
        "__val",
        index_node,
        None,
        dace.Memlet(data=index_data, subset="0"),
    )

    input_edges = [
        gtir_dataflow.EmptyInputEdge(state, index_write_tasklet),
    ]
    output_edge = gtir_dataflow.DataflowOutputEdge(state, index_value)
    return _create_field_operator(
        sdfg, state, domain, node.type, sdfg_builder, input_edges, output_edge
    )


def _get_data_nodes(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    data_name: str,
    data_type: ts.DataType,
) -> FieldopResult:
    if isinstance(data_type, ts.FieldType):
        data_node = state.add_access(data_name)
        return sdfg_builder.make_field(data_node, data_type)

    elif isinstance(data_type, ts.ScalarType):
        if data_name in sdfg.symbols:
            data_node = _get_symbolic_value(
                sdfg, state, sdfg_builder, data_name, data_type, temp_name=f"__{data_name}"
            )
        else:
            data_node = state.add_access(data_name)
        return sdfg_builder.make_field(data_node, data_type)

    elif isinstance(data_type, ts.TupleType):
        symbol_tree = dace_gtir_utils.make_symbol_tree(data_name, data_type)
        return gtx_utils.tree_map(
            lambda sym: _get_data_nodes(sdfg, state, sdfg_builder, sym.id, sym.type)
        )(symbol_tree)

    else:
        raise NotImplementedError(f"Symbol type {type(data_type)} not supported.")


def _get_symbolic_value(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
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
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    """Generates the dataflow subgraph for a `ir.Literal` node."""
    assert isinstance(node, gtir.Literal)

    data_type = node.type
    data_node = _get_symbolic_value(sdfg, state, sdfg_builder, node.value, data_type)

    return FieldopData(data_node, data_type, offset=None)


def translate_make_tuple(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    assert cpm.is_call_to(node, "make_tuple")
    return tuple(
        sdfg_builder.visit(
            arg,
            sdfg=sdfg,
            head_state=state,
        )
        for arg in node.args
    )


def translate_tuple_get(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    assert cpm.is_call_to(node, "tuple_get")
    assert len(node.args) == 2

    if not isinstance(node.args[0], gtir.Literal):
        raise ValueError("Tuple can only be subscripted with compile-time constants.")
    assert ti.is_integral(node.args[0].type)
    index = int(node.args[0].value)

    data_nodes = sdfg_builder.visit(
        node.args[1],
        sdfg=sdfg,
        head_state=state,
    )
    if isinstance(data_nodes, FieldopData):
        raise ValueError(f"Invalid tuple expression {node}")
    unused_arg_nodes: Iterable[FieldopData] = gtx_utils.flatten_nested_tuple(
        tuple(arg for i, arg in enumerate(data_nodes) if i != index)
    )
    state.remove_nodes_from(
        [arg.dc_node for arg in unused_arg_nodes if state.degree(arg.dc_node) == 0]
    )
    return data_nodes[index]


def translate_scalar_expr(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    assert isinstance(node, gtir.FunCall)
    assert isinstance(node.type, ts.ScalarType)

    args = []
    connectors = []
    scalar_expr_args = []

    for i, arg_expr in enumerate(node.args):
        visit_expr = True
        if isinstance(arg_expr, gtir.SymRef):
            try:
                # check if symbol is defined in the GT4Py program, throws `KeyError` exception if undefined
                sdfg_builder.get_symbol_type(arg_expr.id)
            except KeyError:
                # all `SymRef` should refer to symbols defined in the program, except in case of non-variable argument,
                # e.g. the type name `float64` used in casting expressions like `cast_(variable, float64)`
                visit_expr = False

        if visit_expr:
            # we visit the argument expression and obtain the access node to
            # a scalar data container, which will be connected to the tasklet
            arg = sdfg_builder.visit(
                arg_expr,
                sdfg=sdfg,
                head_state=state,
            )
            if not (isinstance(arg, FieldopData) and isinstance(node.type, ts.ScalarType)):
                raise ValueError(f"Invalid argument to scalar expression {arg_expr}.")
            param = f"__arg{i}"
            args.append(arg.dc_node)
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
    temp_name, _ = sdfg_builder.add_temp_scalar(sdfg, dace_utils.as_dace_type(node.type))
    temp_node = state.add_access(temp_name)
    state.add_edge(
        tasklet_node,
        "__out",
        temp_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )

    return FieldopData(temp_node, node.type, offset=None)


def translate_scan(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
) -> FieldopResult:
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # parse the domain of the scan field operator
    domain = extract_domain(domain_expr)

    # use the vertical dimension in the domain as scan dimension
    scan_domain = [
        (dim, lower_bound, upper_bound)
        for dim, lower_bound, upper_bound in domain
        if sdfg_builder.is_column_axis(dim)
    ]
    assert len(scan_domain) == 1
    scan_dim, scan_lower_bound, scan_upper_bound = scan_domain[0]

    # parse scan parameters
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # params[0]: the lambda parameter to propagate the scan carry on the vertical dimension
    scan_carry = str(stencil_expr.params[0].id)

    # params[1]: boolean flag for forward/backward scan
    assert isinstance(scan_expr.args[1], gtir.Literal) and ti.is_logical(scan_expr.args[1].type)
    scan_forward = scan_expr.args[1].value == "True"

    # params[2]: the expression that computes the value for scan initialization
    init_expr = scan_expr.args[2]
    # visit the initialization value of the scan expression
    init_data = sdfg_builder.visit(init_expr, sdfg=sdfg, head_state=state)
    # extract type definition of the scan carry
    scan_carry_type = (
        init_data.gt_type if isinstance(init_data, FieldopData) else get_tuple_type(init_data)
    )

    # make naming consistent throughut this function scope
    def scan_input_name(input_name: str) -> str:
        return f"__gtir_scan_input_{input_name}"

    def scan_output_name(input_name: str) -> str:
        return f"__gtir_scan_output_{input_name}"

    # define the set of symbols available in the lambda context, which consists of
    # the carry argument and all lambda function arguments
    lambda_arg_types = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_symbols = {
        str(p.id): arg_type
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    }

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, sdfg=sdfg, head_state=state) for arg in node.args]
    lambda_args_mapping = {
        scan_input_name(scan_carry): init_data,
    } | {
        str(param.id): arg for param, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)
    }

    # parse the dataflow input and output symbols
    lambda_flat_args: dict[str, FieldopData] = {}
    # the field offset is set to `None` when it is zero in all dimensions
    lambda_field_offsets: dict[str, Optional[list[dace.symbolic.SymExpr]]] = {}
    for param, outer_arg in lambda_args_mapping.items():
        tuple_fields = flatten_tuples(param, outer_arg)
        lambda_field_offsets |= {tsym: tfield.offset for tsym, tfield in tuple_fields}
        lambda_flat_args |= dict(tuple_fields)
    if isinstance(scan_carry_type, ts.TupleType):
        lambda_flat_outs = {
            str(sym.id): sym.type
            for sym in dace_gtir_utils.flatten_tuple_fields(
                scan_output_name(scan_carry), scan_carry_type
            )
        }
    else:
        lambda_flat_outs = {scan_output_name(scan_carry): scan_carry_type}

    # the lambda expression, i.e. body of the scan, will be created inside a nested SDFG.
    nsdfg = dace.SDFG(sdfg_builder.unique_nsdfg_name(sdfg, "scan"))
    nsdfg.debuginfo = dace_utils.debug_info(node, default=sdfg.debuginfo)
    lambda_translator = sdfg_builder.nested_context(
        stencil_expr, nsdfg, lambda_symbols, lambda_field_offsets
    )

    # in case the scan operator computes a list (not a scalar), we need to add an extra dimension
    def get_scan_output_shape(scan_init_data: FieldopData) -> list[dace.symbolic.SymExpr]:
        scan_column_size = scan_upper_bound - scan_lower_bound
        if isinstance(scan_init_data.gt_type, ts.ScalarType):
            return [scan_column_size]
        assert isinstance(scan_init_data.gt_type, ts.ListType)
        assert scan_init_data.gt_type.offset_type
        offset_type = scan_init_data.gt_type.offset_type
        offset_provider_type = sdfg_builder.get_offset_provider_type(offset_type.value)
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
        list_size = offset_provider_type.max_neighbors
        return [scan_column_size, dace.symbolic.SymExpr(list_size)]

    if isinstance(init_data, tuple):
        lambda_result_shape = gtx_utils.tree_map(get_scan_output_shape)(init_data)
    else:
        lambda_result_shape = get_scan_output_shape(init_data)

    # extract the scan loop range
    scan_loop_var = dace_gtir_utils.get_map_variable(scan_dim)

    # create a loop region for lambda call over the scan dimension
    if scan_forward:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} < {scan_upper_bound}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_lower_bound}",
            update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
            inverted=False,
        )
    else:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} >= {scan_lower_bound}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {scan_upper_bound} - 1",
            update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
            inverted=False,
        )

    nsdfg.add_node(scan_loop)
    init_state = nsdfg.add_state("scan_init", is_start_block=True)
    nsdfg.add_edge(init_state, scan_loop, dace.InterstateEdge())
    compute_state = scan_loop.add_state("scan_compute")
    update_state = scan_loop.add_state_after(compute_state, "scan_update")

    # visit the list of arguments to be passed to the scan expression
    stencil_args = [
        _parse_fieldop_arg(
            im.ref(p.id), nsdfg, compute_state, lambda_translator, domain, use_full_shape=True
        )
        for p in stencil_expr.params
    ]

    # generate the dataflow representing the scan field operator
    lambda_input_edges, lambda_result = gtir_dataflow.translate_lambda_to_dataflow(
        nsdfg, compute_state, lambda_translator, stencil_expr, args=stencil_args
    )

    # now initialize the scan carry
    scan_carry_input = (
        dace_gtir_utils.make_symbol_tree(scan_carry, scan_carry_type)
        if isinstance(scan_carry_type, ts.TupleType)
        else im.sym(scan_carry, scan_carry_type)
    )

    def init_scan_carry(sym: gtir.Sym) -> None:
        scan_carry_dataname = str(sym.id)
        scan_carry_desc = nsdfg.data(scan_carry_dataname)
        input_scan_carry_dataname = scan_input_name(scan_carry_dataname)
        input_scan_carry_desc = scan_carry_desc.clone()
        nsdfg.add_datadesc(input_scan_carry_dataname, input_scan_carry_desc)
        scan_carry_desc.transient = True
        init_state.add_nedge(
            init_state.add_access(input_scan_carry_dataname),
            init_state.add_access(scan_carry_dataname),
            nsdfg.make_array_memlet(input_scan_carry_dataname),
        )

    if isinstance(scan_carry_input, tuple):
        gtx_utils.tree_map(init_scan_carry)(scan_carry_input)
    else:
        init_scan_carry(scan_carry_input)

    # connect the dataflow input directly to the source data nodes, without passing through a map node;
    # the reason is that the map for horizontal domain is outside the scan loop region
    for edge in lambda_input_edges:
        edge.connect(map_entry=None)

    # connect the dataflow result nodes to the carry variables
    output_column_index = dace.symbolic.pystr_to_symbolic(scan_loop_var) - scan_lower_bound

    def connect_scan_output(
        scan_output_edge: gtir_dataflow.DataflowOutputEdge,
        scan_output_shape: list[dace.symbolic.SymExpr],
        sym: gtir.Sym,
    ) -> FieldopData:
        scan_result = scan_output_edge.result
        if isinstance(scan_result.gt_dtype, ts.ScalarType):
            assert scan_result.gt_dtype == sym.type
            # the scan field operator computes a column of scalar values
            assert len(scan_output_shape) == 1
            output_subset = dace_subsets.Range.from_string(str(output_column_index))
        else:
            assert isinstance(sym.type, ts.ListType)
            assert scan_result.gt_dtype.element_type == sym.type.element_type
            # the scan field operator computes a list of scalar values for each column level
            assert len(scan_output_shape) == 2
            output_subset = dace_subsets.Range.from_string(
                f"{output_column_index}, 0:{scan_output_shape[1]}"
            )
        scan_result_data = scan_result.dc_node.data
        scan_result_desc = scan_result.dc_node.desc(nsdfg)

        # `sym` represents the global output data, that is the nested-SDFG output connector
        lambda_output = str(sym.id)
        output = scan_output_name(lambda_output)
        nsdfg.add_array(output, scan_output_shape, scan_result_desc.dtype)
        output_node = compute_state.add_access(output)

        compute_state.add_nedge(
            scan_result.dc_node, output_node, dace.Memlet(data=output, subset=output_subset)
        )

        update_state.add_nedge(
            update_state.add_access(scan_result_data),
            update_state.add_access(lambda_output),
            dace.Memlet(data=lambda_output, subset="0"),
        )

        output_type = ts.FieldType(dims=[scan_dim], dtype=scan_result.gt_dtype)
        return FieldopData(output_node, output_type, offset=scan_lower_bound)

    if isinstance(scan_carry_input, tuple):
        assert isinstance(lambda_result, tuple)
        assert isinstance(lambda_result_shape, tuple)
        lambda_output = gtx_utils.tree_map(connect_scan_output)(
            lambda_result, lambda_result_shape, scan_carry_input
        )
    else:
        assert isinstance(lambda_result, gtir_dataflow.DataflowOutputEdge)
        assert isinstance(lambda_result_shape, list)
        lambda_output = connect_scan_output(lambda_result, lambda_result_shape, scan_carry_input)

    # in case tuples are passed as argument, isolated access nodes might be left in the state,
    # because not all tuple fields are necessarily accessed inside the lambda scope
    for data_node in compute_state.data_nodes():
        data_desc = data_node.desc(nsdfg)
        if (compute_state.degree(data_node) == 0) and (
            (not data_desc.transient)
            or data_node.data.startswith(
                scan_carry
            )  # exceptional case where the carry variable is not used, not a scan indeed
        ):
            compute_state.remove_node(data_node)

    # build the mapping of symbols from nested SDFG to parent SDFG
    nsdfg_symbols_mapping = {str(sym): sym for sym in nsdfg.free_symbols}
    for inner_dataname, outer_arg in lambda_flat_args.items():
        inner_desc = nsdfg.data(inner_dataname)
        outer_desc = outer_arg.dc_node.desc(sdfg)
        nsdfg_symbols_mapping |= {
            str(nested_symbol): parent_symbol
            for nested_symbol, parent_symbol in zip(
                [*inner_desc.shape, *inner_desc.strides],
                [*outer_desc.shape, *outer_desc.strides],
                strict=True,
            )
            if isinstance(nested_symbol, dace.symbol)
        }

    # the scan nested SDFG is ready, now we need to instantiate it inside the map implementing the field operator
    nsdfg_node = state.add_nested_sdfg(
        nsdfg,
        sdfg,
        inputs=set(lambda_flat_args.keys()),
        outputs=set(lambda_flat_outs.keys()),
        symbol_mapping=nsdfg_symbols_mapping,
    )

    lambda_input_edges = []
    for input_connector, outer_arg in lambda_flat_args.items():
        arg_desc = outer_arg.dc_node.desc(sdfg)
        input_subset = dace_subsets.Range.from_array(arg_desc)
        input_edge = gtir_dataflow.MemletInputEdge(
            state, outer_arg.dc_node, input_subset, nsdfg_node, input_connector
        )
        lambda_input_edges.append(input_edge)

    def construct_output_edge(scan_data: FieldopData) -> gtir_dataflow.DataflowOutputEdge:
        assert isinstance(scan_data.gt_type, ts.FieldType)
        inner_data = scan_data.dc_node.data
        inner_desc = nsdfg.data(inner_data)
        output_data, output_desc = sdfg.add_temp_transient_like(inner_desc)
        output_node = state.add_access(output_data)
        state.add_edge(
            nsdfg_node,
            inner_data,
            output_node,
            None,
            dace.Memlet.from_array(output_data, output_desc),
        )
        output_expr = gtir_dataflow.ValueExpr(output_node, scan_data.gt_type.dtype)
        return gtir_dataflow.DataflowOutputEdge(state, output_expr)

    output_edges = (
        construct_output_edge(lambda_output)
        if isinstance(lambda_output, FieldopData)
        else gtx_utils.tree_map(construct_output_edge)(lambda_output)
    )

    return _create_field_operator(
        sdfg, state, domain, node.type, sdfg_builder, lambda_input_edges, output_edges, scan_dim
    )


def translate_symbol_ref(
    node: gtir.Node,
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
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
        translate_as_fieldop,
        translate_if,
        translate_index,
        translate_literal,
        translate_make_tuple,
        translate_tuple_get,
        translate_scalar_expr,
        translate_scan,
        translate_symbol_ref,
    ]
