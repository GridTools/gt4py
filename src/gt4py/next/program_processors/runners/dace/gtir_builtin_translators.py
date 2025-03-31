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
from gt4py.next.program_processors.runners.dace import (
    gtir_dataflow,
    gtir_python_codegen,
    gtir_sdfg,
    gtir_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.program_processors.runners.dace.gtir_scan_translator import translate_scan
from gt4py.next.type_system import type_info as ti, type_specifications as ts


if TYPE_CHECKING:
    from gt4py.next.program_processors.runners.dace import gtir_sdfg


def get_domain_indices(
    dims: Sequence[gtx_common.Dimension], origin: Optional[Sequence[dace.symbolic.SymExpr]]
) -> dace_subsets.Indices:
    """
    Helper function to construct the list of indices for a field domain, applying
    an optional origin in each dimension as start index.

    Args:
        dims: The field dimensions.
        origin: The domain start index in each dimension. If set to `None`, assume all zeros.

    Returns:
        A list of indices for field access in dace arrays. As this list is returned
        as `dace.subsets.Indices`, it should be converted to `dace.subsets.Range` before
        being used in memlet subset because ranges are better supported throughout DaCe.
    """
    assert len(dims) != 0
    index_variables = [
        dace.symbolic.pystr_to_symbolic(gtir_sdfg_utils.get_map_variable(dim)) for dim in dims
    ]
    origin = [0] * len(index_variables) if origin is None else origin
    return dace_subsets.Indices(
        [index - start_index for index, start_index in zip(index_variables, origin, strict=True)]
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
        origin: Tuple of start indices, in each dimension, for `FieldType` data.
            Pass an empty tuple for `ScalarType` data or zero-dimensional fields.
    """

    dc_node: dace.nodes.AccessNode
    gt_type: ts.FieldType | ts.ScalarType
    origin: tuple[dace.symbolic.SymbolicType, ...]

    def __post_init__(self) -> None:
        """Implements a sanity check on the constructed data type."""
        assert (
            len(self.origin) == 0
            if isinstance(self.gt_type, ts.ScalarType)
            else len(self.origin) == len(self.gt_type.dims)
        )

    def map_to_parent_sdfg(
        self,
        sdfg_builder: gtir_sdfg.SDFGBuilder,
        inner_sdfg: dace.SDFG,
        outer_sdfg: dace.SDFG,
        outer_sdfg_state: dace.SDFGState,
        symbol_mapping: dict[str, dace.symbolic.SymbolicType],
    ) -> FieldopData:
        """
        Make the data descriptor which 'self' refers to, and which is located inside
        a NestedSDFG, available in its parent SDFG.

        Thus, it turns 'self' into a non-transient array and creates a new data
        descriptor inside the parent SDFG, with same shape and strides.
        """
        inner_desc = self.dc_node.desc(inner_sdfg)
        assert inner_desc.transient
        inner_desc.transient = False

        if isinstance(self.gt_type, ts.ScalarType):
            outer, outer_desc = sdfg_builder.add_temp_scalar(outer_sdfg, inner_desc.dtype)
            outer_origin = []
        else:
            outer, outer_desc = sdfg_builder.add_temp_array_like(outer_sdfg, inner_desc)
            # We cannot use a copy of the inner data descriptor directly, we have to apply the symbol mapping.
            dace.symbolic.safe_replace(
                symbol_mapping,
                lambda m: dace.sdfg.replace_properties_dict(outer_desc, m),
            )
            # Same applies to the symbols used as field origin (the domain range start)
            outer_origin = [
                gtx_dace_utils.safe_replace_symbolic(val, symbol_mapping) for val in self.origin
            ]

        outer_node = outer_sdfg_state.add_access(outer)
        return FieldopData(outer_node, self.gt_type, tuple(outer_origin))

    def get_local_view(
        self, domain: FieldopDomain, sdfg: dace.SDFG
    ) -> gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr:
        """Helper method to access a field in local view, given the compute domain of a field operator."""
        if isinstance(self.gt_type, ts.ScalarType):
            assert isinstance(self.dc_node.desc(sdfg), dace.data.Scalar)
            return gtir_dataflow.MemletExpr(
                dc_node=self.dc_node,
                gt_dtype=self.gt_type,
                subset=dace_subsets.Range.from_string("0"),
            )

        if isinstance(self.gt_type, ts.FieldType):
            it_indices: dict[gtx_common.Dimension, gtir_dataflow.DataExpr]
            if isinstance(self.dc_node.desc(sdfg), dace.data.Scalar):
                assert len(self.gt_type.dims) == 0  # zero-dimensional field
                it_indices = {}
            else:
                # The invariant below is ensured by calling `make_field()` to construct `FieldopData`.
                # The `make_field` constructor converts any local dimension, if present, to `ListType`
                # element type, while leaving the field domain with all global dimensions.
                assert all(dim != gtx_common.DimensionKind.LOCAL for dim in self.gt_type.dims)
                domain_dims = [dim for dim, _, _ in domain]
                domain_indices = get_domain_indices(domain_dims, origin=None)
                it_indices = {
                    dim: gtir_dataflow.SymbolExpr(index, INDEX_DTYPE)
                    for dim, index in zip(domain_dims, domain_indices)
                }
            field_origin = [
                (dim, dace.symbolic.SymExpr(0) if self.origin is None else self.origin[i])
                for i, dim in enumerate(self.gt_type.dims)
            ]
            return gtir_dataflow.IteratorExpr(
                self.dc_node, self.gt_type.dtype, field_origin, it_indices
            )

        raise NotImplementedError(f"Node type {type(self.gt_type)} not supported.")

    def get_symbol_mapping(
        self, dataname: str, sdfg: dace.SDFG
    ) -> dict[str, dace.symbolic.SymExpr]:
        """
        Helper method to create the symbol mapping for array storage in a nested SDFG.

        Args:
            dataname: Name of the data container insiode the nested SDFG.
            sdfg: The parent SDFG where the `FieldopData` object lives.

        Returns:
            Mapping from symbols in nested SDFG to the corresponding symbolic values
            in the parent SDFG. This includes the range start and stop symbols (used
            to calculate the array shape as range 'stop - start') and the strides.
        """
        if isinstance(self.gt_type, ts.ScalarType):
            return {}
        ndims = len(self.gt_type.dims)
        outer_desc = self.dc_node.desc(sdfg)
        assert isinstance(outer_desc, dace.data.Array)
        # origin and size of the local dimension, in case of a field with `ListType` data,
        # are assumed to be compiled-time values (not symbolic), therefore the start and
        # stop range symbols of the inner field only extend over the global dimensions
        return (
            {gtx_dace_utils.range_start_symbol(dataname, i): (self.origin[i]) for i in range(ndims)}
            | {
                gtx_dace_utils.range_stop_symbol(dataname, i): (
                    self.origin[i] + outer_desc.shape[i]
                )
                for i in range(ndims)
            }
            | {
                gtx_dace_utils.field_stride_symbol_name(dataname, i): stride
                for i, stride in enumerate(outer_desc.strides)
            }
        )


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


def get_arg_symbol_mapping(
    dataname: str, arg: FieldopResult, sdfg: dace.SDFG
) -> dict[str, dace.symbolic.SymExpr]:
    """
    Helper method to build the mapping from inner to outer SDFG of all symbols
    used for storage of a field or a tuple of fields.

    Args:
        dataname: The storage name inside the nested SDFG.
        arg: The argument field in the parent SDFG.
        sdfg: The parent SDFG where the argument field lives.

    Returns:
        A mapping from inner symbol names to values or symbolic definitions
        in the parent SDFG.
    """
    if isinstance(arg, FieldopData):
        return arg.get_symbol_mapping(dataname, sdfg)

    symbol_mapping: dict[str, dace.symbolic.SymExpr] = {}
    for i, elem in enumerate(arg):
        dataname_elem = f"{dataname}_{i}"
        symbol_mapping |= get_arg_symbol_mapping(dataname_elem, elem, sdfg)

    return symbol_mapping


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
        tuple_symbols = gtir_sdfg_utils.flatten_tuple_fields(name, tuple_type)
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
) -> (
    gtir_dataflow.IteratorExpr
    | gtir_dataflow.MemletExpr
    | tuple[gtir_dataflow.IteratorExpr | gtir_dataflow.MemletExpr | tuple[Any, ...], ...]
):
    """Helper method to visit an expression passed as argument to a field operator."""

    arg = sdfg_builder.visit(node, sdfg=sdfg, head_state=state)

    if isinstance(arg, FieldopData):
        return arg.get_local_view(domain, sdfg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(lambda targ: targ.get_local_view(domain))(arg)


def get_field_layout(
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
            - the domain origin, that is the start indices in all dimensions
            - the domain size in each dimension
    """
    if len(domain) == 0:
        return [], [], []
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
    Helper method to allocate a temporary array that stores one field computed
    by a field operator.

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
        The field data descriptor, which includes the field access node in the
        given `state` and the field domain offset.
    """
    dataflow_output_desc = output_edge.result.dc_node.desc(sdfg)

    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, field_shape = get_field_layout(domain)
    if len(domain) == 0:
        # The field operator computes a zero-dimensional field, and the data subset
        # is set later depending on the element type (`ts.ListType` or `ts.ScalarType`)
        field_subset = dace_subsets.Range([])
    else:
        field_indices = get_domain_indices(field_dims, field_origin)
        field_subset = dace_subsets.Range.from_indices(field_indices)

    if isinstance(output_edge.result.gt_dtype, ts.ScalarType):
        if output_edge.result.gt_dtype != output_type.dtype:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype} got {output_edge.result.gt_dtype}."
            )
        assert isinstance(dataflow_output_desc, dace.data.Scalar)
    else:
        assert isinstance(output_type.dtype, ts.ListType)
        assert isinstance(output_edge.result.gt_dtype.element_type, ts.ScalarType)
        if output_edge.result.gt_dtype.element_type != output_type.dtype.element_type:
            raise TypeError(
                f"Type mismatch, expected {output_type.dtype.element_type} got {output_edge.result.gt_dtype.element_type}."
            )
        assert isinstance(dataflow_output_desc, dace.data.Array)
        assert len(dataflow_output_desc.shape) == 1
        # extend the array with the local dimensions added by the field operator (e.g. `neighbors`)
        assert output_edge.result.gt_dtype.offset_type is not None
        field_shape = [*field_shape, dataflow_output_desc.shape[0]]
        field_subset = field_subset + dace_subsets.Range.from_array(dataflow_output_desc)

    # allocate local temporary storage
    if len(field_shape) == 0:  # zero-dimensional field
        field_name, _ = sdfg_builder.add_temp_scalar(sdfg, dataflow_output_desc.dtype)
        field_subset = dace_subsets.Range.from_string("0")
    else:
        field_name, _ = sdfg_builder.add_temp_array(sdfg, field_shape, dataflow_output_desc.dtype)
    field_node = state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    output_edge.connect(map_exit, field_node, field_subset)

    return FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def _create_field_operator(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    domain: FieldopDomain,
    node_type: ts.FieldType | ts.TupleType,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    input_edges: Iterable[gtir_dataflow.DataflowInputEdge],
    output_tree: tuple[gtir_dataflow.DataflowOutputEdge | tuple[Any, ...], ...],
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
        output_tree: A tree representation of the dataflow output data.

    Returns:
        The descriptor of the field operator result, which can be either a single
        field or a tuple fields.
    """

    if len(domain) == 0:
        # create a trivial map for zero-dimensional fields
        map_range = {
            "__gt4py_zerodim": "0",
        }
    else:
        # create map range corresponding to the field operator domain
        map_range = {
            gtir_sdfg_utils.get_map_variable(dim): f"{lower_bound}:{upper_bound}"
            for dim, lower_bound, upper_bound in domain
        }
    map_entry, map_exit = sdfg_builder.add_map("fieldop", state, map_range)

    # here we setup the edges passing through the map entry node
    for edge in input_edges:
        edge.connect(map_entry)

    if isinstance(node_type, ts.FieldType):
        assert len(output_tree) == 1 and isinstance(
            output_tree[0], gtir_dataflow.DataflowOutputEdge
        )
        output_edge = output_tree[0]
        return _create_field_operator_impl(
            sdfg_builder, sdfg, state, domain, output_edge, node_type, map_exit
        )
    else:
        # handle tuples of fields
        output_symbol_tree = gtir_sdfg_utils.make_symbol_tree("x", node_type)
        return gtx_utils.tree_map(
            lambda output_edge, output_sym: _create_field_operator_impl(
                sdfg_builder, sdfg, state, domain, output_edge, output_sym.type, map_exit
            )
        )(output_tree, output_symbol_tree)


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


def _construct_if_branch_output(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.SDFGBuilder,
    domain: gtir.Expr,
    sym: gtir.Sym,
    true_br: FieldopData,
    false_br: FieldopData,
) -> FieldopData:
    """
    Helper function called by `translate_if()` to allocate a temporary field to store
    the result of an if expression.
    """
    assert true_br.gt_type == false_br.gt_type
    out_type = true_br.gt_type

    if isinstance(sym.type, ts.ScalarType):
        assert sym.type == out_type
        dtype = gtx_dace_utils.as_dace_type(sym.type)
        out, _ = sdfg_builder.add_temp_scalar(sdfg, dtype)
        out_node = state.add_access(out)
        return FieldopData(out_node, sym.type, origin=())

    assert isinstance(out_type, ts.FieldType)
    assert isinstance(sym.type, ts.FieldType)
    dims, origin, shape = get_field_layout(extract_domain(domain))
    assert dims == out_type.dims

    if isinstance(out_type.dtype, ts.ScalarType):
        dtype = gtx_dace_utils.as_dace_type(out_type.dtype)
    else:
        assert isinstance(out_type.dtype, ts.ListType)
        assert out_type.dtype.offset_type is not None
        assert isinstance(out_type.dtype.element_type, ts.ScalarType)
        dtype = gtx_dace_utils.as_dace_type(out_type.dtype.element_type)
        offset_provider_type = sdfg_builder.get_offset_provider_type(
            out_type.dtype.offset_type.value
        )
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
        shape = [*shape, offset_provider_type.max_neighbors]

    out, _ = sdfg_builder.add_temp_array(sdfg, shape, dtype)
    out_node = state.add_access(out)

    return FieldopData(out_node, out_type, tuple(origin))


def _write_if_branch_output(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    src: FieldopData,
    dst: FieldopData,
) -> None:
    """
    Helper function called by `translate_if()` to write the result of an if-branch,
    here `src` field, to the output 'dst' field. The data subset is based on the
    domain of the `dst` field. Therefore, the full shape of `dst` array is written.
    """
    if src.gt_type != dst.gt_type:
        raise ValueError(
            f"Source and destination type mismatch, '{dst.gt_type}' vs '{src.gt_type}'."
        )
    dst_node = state.add_access(dst.dc_node.data)
    dst_shape = dst_node.desc(sdfg).shape

    if isinstance(src.gt_type, ts.ScalarType):
        state.add_nedge(
            src.dc_node,
            dst_node,
            dace.Memlet(data=src.dc_node.data, subset="0"),
        )
    else:
        if isinstance(src.gt_type.dtype, ts.ListType):
            src_origin = [*src.origin, 0]
            dst_origin = [*dst.origin, 0]
        else:
            src_origin = [*src.origin]
            dst_origin = [*dst.origin]

        data_subset = dace_subsets.Range(
            (
                f"{dst_start - src_start}",
                f"{dst_start - src_start + size - 1}",  # subtract 1 because the range boundaries are included
                1,
            )
            for src_start, dst_start, size in zip(src_origin, dst_origin, dst_shape, strict=True)
        )

        state.add_nedge(
            src.dc_node,
            dst_node,
            dace.Memlet(
                data=src.dc_node.data,
                subset=data_subset,
                other_subset=dace_subsets.Range.from_array(dst_node.desc(sdfg)),
            ),
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

    # evaluate the if-condition in a new entry state and use the current head state
    # to join the true/false branch states as follows:
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

    true_br_result = sdfg_builder.visit(
        true_expr,
        sdfg=sdfg,
        head_state=true_state,
    )
    false_br_result = sdfg_builder.visit(
        false_expr,
        sdfg=sdfg,
        head_state=false_state,
    )

    if isinstance(node.type, ts.TupleType):
        symbol_tree = gtir_sdfg_utils.make_symbol_tree("x", node.type)
        if isinstance(node.annex.domain, tuple):
            domain_tree = node.annex.domain
        else:
            # TODO(edopao): this is a workaround for some IR nodes where the inferred
            #   domain on a tuple of fields is not a tuple, see `test_execution.py::test_ternary_operator_tuple()`
            domain_tree = gtx_utils.tree_map(lambda _: node.annex.domain)(symbol_tree)
        node_output = gtx_utils.tree_map(
            lambda sym,
            domain,
            true_br,
            false_br,
            sdfg=sdfg,
            state=state,
            sdfg_builder=sdfg_builder: _construct_if_branch_output(
                sdfg,
                state,
                sdfg_builder,
                domain,
                sym,
                true_br,
                false_br,
            )
        )(
            symbol_tree,
            domain_tree,
            true_br_result,
            false_br_result,
        )
        gtx_utils.tree_map(
            lambda src, dst, state=true_state: _write_if_branch_output(sdfg, state, src, dst)
        )(true_br_result, node_output)
        gtx_utils.tree_map(
            lambda src, dst, state=false_state: _write_if_branch_output(sdfg, state, src, dst)
        )(false_br_result, node_output)
    else:
        node_output = _construct_if_branch_output(
            sdfg,
            state,
            sdfg_builder,
            node.annex.domain,
            im.sym("x", node.type),
            true_br_result,
            false_br_result,
        )
        _write_if_branch_output(sdfg, true_state, true_br_result, node_output)
        _write_if_branch_output(sdfg, false_state, false_br_result, node_output)

    return node_output


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
    dim_index = gtir_sdfg_utils.get_map_variable(dim)

    index_data, _ = sdfg_builder.add_temp_scalar(sdfg, INDEX_DTYPE)
    index_node = state.add_access(index_data)
    index_value = gtir_dataflow.ValueExpr(
        dc_node=index_node,
        gt_dtype=gtx_dace_utils.as_itir_type(INDEX_DTYPE),
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
        sdfg, state, domain, node.type, sdfg_builder, input_edges, (output_edge,)
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
        return FieldopData(data_node, data_type, origin=())

    elif isinstance(data_type, ts.TupleType):
        symbol_tree = gtir_sdfg_utils.make_symbol_tree(data_name, data_type)
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
        gtx_dace_utils.as_dace_type(scalar_type),
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

    return FieldopData(data_node, data_type, origin=())


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
    temp_name, _ = sdfg_builder.add_temp_scalar(sdfg, gtx_dace_utils.as_dace_type(node.type))
    temp_node = state.add_access(temp_name)
    state.add_edge(
        tasklet_node,
        "__out",
        temp_node,
        None,
        dace.Memlet(data=temp_name, subset="0"),
    )

    return FieldopData(temp_node, node.type, origin=())


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
