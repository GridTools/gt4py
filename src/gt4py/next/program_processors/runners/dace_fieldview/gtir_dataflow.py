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
from typing import Any, Dict, Final, List, Optional, Protocol, Set, Tuple, TypeAlias, Union

import dace
import dace.subsets as sbs

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import type_specifications as itir_ts
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_python_codegen,
    gtir_sdfg,
    utility as dace_gtir_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


# Magic local dimension for the result of a `make_const_list`.
# A clean implementation will probably involve to tag the `make_const_list`
# with the neighborhood it is meant to be used with.
_CONST_DIM = gtx_common.Dimension(value="_CONST_DIM", kind=gtx_common.DimensionKind.LOCAL)


@dataclasses.dataclass(frozen=True)
class ValueExpr:
    """
    Local storage for the values returned by dataflow computation.

    This type is used in the context in a dataflow, that is a stencil expression.
    Therefore, it contains either a scalar value (single elements in the fields) or
    a list of values in a local dimension.
    This is different from `gtir_builtin_translators.FieldopData` which represents
    the result of a field operator, basically the data storage outside a global map.

    Args:
        dc_node: Access node to the data container, can be either a scalar or a local list.
        gt_dtype: GT4Py data type, which includes the `offset_type` local dimension for lists.
    """

    dc_node: dace.nodes.AccessNode
    gt_dtype: itir_ts.ListType | ts.ScalarType


@dataclasses.dataclass(frozen=True)
class MemletExpr:
    """
    Scalar or array data access through a memlet.

    Args:
        dc_node: Access node to the data container, can be either a scalar or a local list.
        gt_dtype: GT4Py data type, which includes the `offset_type` local dimension for lists.
        subset: Represents the subset to use in memlet to access the above data.
    """

    dc_node: dace.nodes.AccessNode
    gt_dtype: itir_ts.ListType | ts.ScalarType
    subset: sbs.Indices | sbs.Range


@dataclasses.dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymExpr
    dc_dtype: dace.typeclass


DataExpr: TypeAlias = ValueExpr | MemletExpr | SymbolExpr


@dataclasses.dataclass(frozen=True)
class IteratorExpr:
    """
    Iterator for field access to be consumed by `deref` or `shift` builtin functions.

    Args:
        field: Access node to the field this iterator operates on.
        gt_dtype: GT4Py data type, which includes the `offset_type` local dimension for lists.
        field_domain: Field domain represented as a sorted list of dimensions and offset values,
            used to find the position of a map index variable in the memlet subset. The offset
            value is either the start index of dimension range or the compile-time value of
            a shift expression, or a composition of both, and it must be subtracted to the index
            variable when constructing the memlet subset range.
        indices: Maps each dimension to an index value, which could be either a symbolic value
            or the result of a tasklet computation like neighbors connectivity or dynamic offset.
    """

    field: dace.nodes.AccessNode
    gt_dtype: itir_ts.ListType | ts.ScalarType
    field_domain: list[tuple[gtx_common.Dimension, dace.symbolic.SymExpr]]
    indices: dict[gtx_common.Dimension, DataExpr]

    def get_memlet_subset(self, sdfg: dace.SDFG) -> sbs.Range:
        if not all(isinstance(self.indices[dim], SymbolExpr) for dim, _ in self.field_domain):
            raise ValueError(f"Cannot deref iterator {self}.")

        field_desc = self.field.desc(sdfg)
        if isinstance(self.gt_dtype, itir_ts.ListType):
            assert len(field_desc.shape) == len(self.field_domain) + 1
            assert self.gt_dtype.offset_type is not None
            field_domain = [*self.field_domain, (self.gt_dtype.offset_type, 0)]
        else:
            assert len(field_desc.shape) == len(self.field_domain)
            field_domain = self.field_domain

        return sbs.Range.from_string(
            ",".join(
                str(self.indices[dim].value - offset)  # type: ignore[union-attr]
                if dim in self.indices
                else f"0:{size}"
                for (dim, offset), size in zip(field_domain, field_desc.shape, strict=True)
            )
        )


class DataflowInputEdge(Protocol):
    """
    This protocol represents an open connection into the dataflow.

    It provides the `connect` method to setup an input edge from an external data source.
    Since the dataflow represents a stencil, we instantiate the dataflow inside a map scope
    and connect its inputs and outputs to external data nodes by means of memlets that
    traverse the map entry and exit nodes.
    """

    @abc.abstractmethod
    def connect(self, me: dace.nodes.MapEntry) -> None: ...


@dataclasses.dataclass(frozen=True)
class MemletInputEdge(DataflowInputEdge):
    """
    Allows to setup an input memlet through a map entry node.

    The edge source has to be a data access node, while the destination node can either
    be a tasklet, in which case the connector name is also required, or an access node.
    """

    state: dace.SDFGState
    source: dace.nodes.AccessNode
    subset: sbs.Range
    dest: dace.nodes.AccessNode | dace.nodes.Tasklet
    dest_conn: Optional[str]

    def connect(self, me: dace.nodes.MapEntry) -> None:
        memlet = dace.Memlet(data=self.source.data, subset=self.subset)
        self.state.add_memlet_path(
            self.source,
            me,
            self.dest,
            dst_conn=self.dest_conn,
            memlet=memlet,
        )


@dataclasses.dataclass(frozen=True)
class EmptyInputEdge(DataflowInputEdge):
    """
    Allows to setup an edge from a map entry node to a tasklet with no arguments.

    The reason behind this kind of connection is that all nodes inside a map scope
    must have an in/out path that traverses the entry and exit nodes.
    """

    state: dace.SDFGState
    node: dace.nodes.Tasklet

    def connect(self, me: dace.nodes.MapEntry) -> None:
        self.state.add_nedge(me, self.node, dace.Memlet())


@dataclasses.dataclass(frozen=True)
class DataflowOutputEdge:
    """
    Allows to setup an output memlet through a map exit node.

    The result of a dataflow subgraph needs to be written to an external data node.
    Since the dataflow represents a stencil and the dataflow is computed over
    a field domain, the dataflow is instatiated inside a map scope. The `connect`
    method creates a memlet that writes the dataflow result to the external array
    passing through the map exit node.
    """

    state: dace.SDFGState
    result: ValueExpr

    def connect(
        self,
        mx: dace.nodes.MapExit,
        dest: dace.nodes.AccessNode,
        subset: sbs.Range,
    ) -> None:
        # retrieve the node which writes the result
        last_node = self.state.in_edges(self.result.dc_node)[0].src
        if isinstance(last_node, dace.nodes.Tasklet):
            # the last transient node can be deleted
            last_node_connector = self.state.in_edges(self.result.dc_node)[0].src_conn
            self.state.remove_node(self.result.dc_node)
        else:
            last_node = self.result.dc_node
            last_node_connector = None

        self.state.add_memlet_path(
            last_node,
            mx,
            dest,
            src_conn=last_node_connector,
            memlet=dace.Memlet(data=dest.data, subset=subset),
        )


DACE_REDUCTION_MAPPING: dict[str, dace.dtypes.ReductionType] = {
    "minimum": dace.dtypes.ReductionType.Min,
    "maximum": dace.dtypes.ReductionType.Max,
    "plus": dace.dtypes.ReductionType.Sum,
    "multiplies": dace.dtypes.ReductionType.Product,
    "and_": dace.dtypes.ReductionType.Logical_And,
    "or_": dace.dtypes.ReductionType.Logical_Or,
    "xor_": dace.dtypes.ReductionType.Logical_Xor,
    "minus": dace.dtypes.ReductionType.Sub,
    "divides": dace.dtypes.ReductionType.Div,
}


def get_reduce_params(node: gtir.FunCall) -> tuple[str, SymbolExpr, SymbolExpr]:
    assert isinstance(node.type, ts.ScalarType)
    dc_dtype = dace_utils.as_dace_type(node.type)

    assert isinstance(node.fun, gtir.FunCall)
    assert len(node.fun.args) == 2
    assert isinstance(node.fun.args[0], gtir.SymRef)
    op_name = str(node.fun.args[0])
    assert isinstance(node.fun.args[1], gtir.Literal)
    assert node.fun.args[1].type == node.type
    reduce_init = SymbolExpr(node.fun.args[1].value, dc_dtype)

    if op_name not in DACE_REDUCTION_MAPPING:
        raise RuntimeError(f"Reduction operation '{op_name}' not supported.")
    identity_value = dace.dtypes.reduction_identity(dc_dtype, DACE_REDUCTION_MAPPING[op_name])
    reduce_identity = SymbolExpr(identity_value, dc_dtype)

    return op_name, reduce_init, reduce_identity


class LambdaToDataflow(eve.NodeVisitor):
    """
    Translates an `ir.Lambda` expression to a dataflow graph.

    The dataflow graph generated here typically represents the stencil function
    of a field operator. It only computes single elements or pure local fields,
    in case of neighbor values. In case of local fields, the dataflow contains
    inner maps with fixed literal size (max number of neighbors).
    Once the lambda expression has been lowered to a dataflow, the dataflow graph
    needs to be instantiated, that is we have to connect all in/out edges to
    external source/destination data nodes. Since the lambda expression is used
    in GTIR as argument to a field operator, the dataflow is instatiated inside
    a map scope and applied on the field domain. Therefore, all in/out edges
    must traverse the entry/exit map nodes.
    """

    sdfg: dace.SDFG
    state: dace.SDFGState
    subgraph_builder: gtir_sdfg.DataflowBuilder
    input_edges: list[DataflowInputEdge]
    symbol_map: dict[str, IteratorExpr | MemletExpr | SymbolExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        subgraph_builder: gtir_sdfg.DataflowBuilder,
    ):
        self.sdfg = sdfg
        self.state = state
        self.subgraph_builder = subgraph_builder
        self.input_edges = []
        self.symbol_map = {}

    def _add_input_data_edge(
        self,
        src: dace.nodes.AccessNode,
        src_subset: sbs.Range,
        dst_node: dace.nodes.Node,
        dst_conn: Optional[str] = None,
        src_offset: Optional[list[dace.symbolic.SymExpr]] = None,
    ) -> None:
        input_subset = (
            src_subset
            if src_offset is None
            else sbs.Range(
                (start - off, stop - off, step)
                for (start, stop, step), off in zip(src_subset, src_offset, strict=True)
            )
        )
        edge = MemletInputEdge(self.state, src, input_subset, dst_node, dst_conn)
        self.input_edges.append(edge)

    def _add_edge(
        self,
        src_node: dace.Node,
        src_node_connector: Optional[str],
        dst_node: dace.Node,
        dst_node_connector: Optional[str],
        memlet: dace.Memlet,
    ) -> None:
        """Helper method to add an edge in current state."""
        self.state.add_edge(src_node, src_node_connector, dst_node, dst_node_connector, memlet)

    def _add_map(
        self,
        name: str,
        ndrange: Union[
            Dict[str, Union[str, dace.subsets.Subset]],
            List[Tuple[str, Union[str, dace.subsets.Subset]]],
        ],
        **kwargs: Any,
    ) -> Tuple[dace.nodes.MapEntry, dace.nodes.MapExit]:
        """
        Helper method to add a map in current state.

        The subgraph builder ensures that the map receives a unique name,
        by adding a unique suffix to the provided name.
        """
        return self.subgraph_builder.add_map(name, self.state, ndrange, **kwargs)

    def _add_tasklet(
        self,
        name: str,
        inputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        code: str,
        **kwargs: Any,
    ) -> dace.nodes.Tasklet:
        """
        Helper method to add a tasklet in current state.

        The subgraph builder ensures that the tasklet receives a unique name,
        by adding a unique suffix to the provided name.
        """
        tasklet_node = self.subgraph_builder.add_tasklet(
            name, self.state, inputs, outputs, code, **kwargs
        )
        if len(inputs) == 0:
            # All nodes inside a map scope must have an in/out path that traverses
            # the entry and exit nodes. Therefore, a tasklet node with no arguments
            # still needs an (empty) input edge from map entry node.
            edge = EmptyInputEdge(self.state, tasklet_node)
            self.input_edges.append(edge)
        return tasklet_node

    def _add_mapped_tasklet(
        self,
        name: str,
        map_ranges: Dict[str, str | dace.subsets.Subset]
        | List[Tuple[str, str | dace.subsets.Subset]],
        inputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        code: str,
        outputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        **kwargs: Any,
    ) -> tuple[dace.nodes.Tasklet, dace.nodes.MapEntry, dace.nodes.MapExit]:
        """
        Helper method to add a mapped tasklet in current state.

        The subgraph builder ensures that the tasklet receives a unique name,
        by adding a unique suffix to the provided name.
        """
        return self.subgraph_builder.add_mapped_tasklet(
            name, self.state, map_ranges, inputs, code, outputs, **kwargs
        )

    def unique_nsdfg_name(self, prefix: str) -> str:
        """Utility function to generate a unique name for a nested SDFG, starting with the given prefix."""
        return self.subgraph_builder.unique_nsdfg_name(self.sdfg, prefix)

    def _construct_local_view(self, field: MemletExpr | ValueExpr) -> ValueExpr:
        if isinstance(field, MemletExpr):
            desc = field.dc_node.desc(self.sdfg)
            local_dim_indices = [i for i, size in enumerate(field.subset.size()) if size != 1]
            if len(local_dim_indices) == 0:
                # we are accessing a single-element array with shape (1,)
                view_shape = (1,)
                view_strides = (1,)
            else:
                view_shape = tuple(desc.shape[i] for i in local_dim_indices)
                view_strides = tuple(desc.strides[i] for i in local_dim_indices)
            view, _ = self.sdfg.add_view(
                f"{field.dc_node.data}_view",
                view_shape,
                desc.dtype,
                strides=view_strides,
                find_new_name=True,
            )
            local_view_node = self.state.add_access(view)
            self._add_input_data_edge(field.dc_node, field.subset, local_view_node)

            return ValueExpr(local_view_node, desc.dtype)

        else:
            return field

    def _construct_tasklet_result(
        self,
        dc_dtype: dace.typeclass,
        src_node: dace.nodes.Tasklet,
        src_connector: str,
        use_array: bool = False,
    ) -> ValueExpr:
        data_type = dace_utils.as_itir_type(dc_dtype)
        if use_array:
            # In some cases, such as result data with list-type annotation, we want
            # that output data is represented as an array (single-element 1D array)
            # in order to allow for composition of array shape in external memlets.
            temp_name, _ = self.sdfg.add_temp_transient((1,), dc_dtype)
        else:
            temp_name = self.sdfg.temp_data_name()
            self.sdfg.add_scalar(temp_name, dc_dtype, transient=True)

        temp_node = self.state.add_access(temp_name)
        self._add_edge(
            src_node,
            src_connector,
            temp_node,
            None,
            dace.Memlet(data=temp_name, subset="0"),
        )
        return ValueExpr(
            dc_node=temp_node,
            gt_dtype=(
                itir_ts.ListType(element_type=data_type, offset_type=_CONST_DIM)
                if use_array
                else data_type
            ),
        )

    def _visit_deref(self, node: gtir.FunCall) -> DataExpr:
        """
        Visit a `deref` node, which represents dereferencing of an iterator.
        The iterator is the argument of this node.

        The iterator contains the information for accessing a field, that is the
        sorted list of dimensions in the field domain and the index values for
        each dimension. The index values can be either symbol values, that is
        literal values or scalar arguments which are constant in the SDFG scope;
        or they can be the result of some expression, that computes a dynamic
        index offset or gets an neighbor index from a connectivity table.
        In case all indexes are symbol values, the `deref` node is lowered to a
        memlet; otherwise dereferencing is a runtime operation represented in
        the SDFG as a tasklet node.
        """
        # format used for field index tasklet connector
        IndexConnectorFmt: Final = "__index_{dim}"

        assert len(node.args) == 1
        arg_expr = self.visit(node.args[0])

        if not isinstance(arg_expr, IteratorExpr):
            # dereferencing a scalar or a literal node results in the node itself
            return arg_expr

        field_desc = arg_expr.field.desc(self.sdfg)
        if isinstance(field_desc, dace.data.Scalar):
            # deref a zero-dimensional field
            assert len(arg_expr.field_domain) == 0
            assert isinstance(node.type, ts.ScalarType)
            return MemletExpr(arg_expr.field, arg_expr.gt_dtype, subset="0")

        # default case: deref a field with one or more dimensions
        if all(isinstance(index, SymbolExpr) for index in arg_expr.indices.values()):
            # when all indices are symbolic expressions, we can perform direct field access through a memlet
            field_subset = arg_expr.get_memlet_subset(self.sdfg)
            return MemletExpr(arg_expr.field, arg_expr.gt_dtype, field_subset)

        # we use a tasklet to dereference an iterator when one or more indices are the result of some computation,
        # either indirection through connectivity table or dynamic cartesian offset.
        assert all(dim in arg_expr.indices for dim, _ in arg_expr.field_domain)
        assert len(field_desc.shape) == len(arg_expr.field_domain)
        field_indices = [(dim, arg_expr.indices[dim]) for dim, _ in arg_expr.field_domain]
        index_connectors = [
            IndexConnectorFmt.format(dim=dim.value)
            for dim, index in field_indices
            if not isinstance(index, SymbolExpr)
        ]
        # here `internals` refer to the names used as index in the tasklet code string:
        # an index can be either a connector name (for dynamic/indirect indices)
        # or a symbol value (for literal values and scalar arguments).
        index_internals = ",".join(
            str(index.value)
            if isinstance(index, SymbolExpr)
            else IndexConnectorFmt.format(dim=dim.value)
            for dim, index in field_indices
        )
        deref_node = self._add_tasklet(
            "runtime_deref",
            {"field"} | set(index_connectors),
            {"val"},
            code=f"val = field[{index_internals}]",
        )
        # add new termination point for the field parameter
        self._add_input_data_edge(
            arg_expr.field,
            sbs.Range.from_array(field_desc),
            deref_node,
            "field",
            src_offset=[offset for (_, offset) in arg_expr.field_domain],
        )

        for dim, index_expr in field_indices:
            # add termination points for the dynamic iterator indices
            deref_connector = IndexConnectorFmt.format(dim=dim.value)
            if isinstance(index_expr, MemletExpr):
                self._add_input_data_edge(
                    index_expr.dc_node,
                    index_expr.subset,
                    deref_node,
                    deref_connector,
                )

            elif isinstance(index_expr, ValueExpr):
                self._add_edge(
                    index_expr.dc_node,
                    None,
                    deref_node,
                    deref_connector,
                    dace.Memlet(data=index_expr.dc_node.data, subset="0"),
                )
            else:
                assert isinstance(index_expr, SymbolExpr)

        return self._construct_tasklet_result(field_desc.dtype, deref_node, "val")

    def _visit_neighbors(self, node: gtir.FunCall) -> ValueExpr:
        assert isinstance(node.type, itir_ts.ListType)
        assert len(node.args) == 2

        assert isinstance(node.args[0], gtir.OffsetLiteral)
        offset = node.args[0].value
        assert isinstance(offset, str)
        offset_provider = self.subgraph_builder.get_offset_provider_type(offset)
        assert isinstance(offset_provider, gtx_common.NeighborConnectivityType)

        it = self.visit(node.args[1])
        assert isinstance(it, IteratorExpr)
        assert any(dim == offset_provider.codomain for dim, _ in it.field_domain)
        assert offset_provider.source_dim in it.indices
        origin_index = it.indices[offset_provider.source_dim]
        assert isinstance(origin_index, SymbolExpr)
        assert all(isinstance(index, SymbolExpr) for index in it.indices.values())

        field_desc = it.field.desc(self.sdfg)
        connectivity = dace_utils.connectivity_identifier(offset)
        # initially, the storage for the connectivty tables is created as transient;
        # when the tables are used, the storage is changed to non-transient,
        # as the corresponding arrays are supposed to be allocated by the SDFG caller
        connectivity_desc = self.sdfg.arrays[connectivity]
        connectivity_desc.transient = False

        # The visitor is constructing a list of input connections that will be handled
        # by `translate_as_fieldop` (the primitive translator), that is responsible
        # of creating the map for the field domain. For each input connection, it will
        # create a memlet that will write to a node specified by the third attribute
        # in the `InputConnection` tuple (either a tasklet, or a view node, or a library
        # node). For the specific case of `neighbors` we need to nest the neighbors map
        # inside the field map and the memlets will traverse the external map and write
        # to the view nodes. The simplify pass will remove the redundant access nodes.
        field_slice = self._construct_local_view(
            MemletExpr(
                dc_node=it.field,
                gt_dtype=node.type,
                subset=sbs.Range.from_string(
                    ",".join(
                        str(it.indices[dim].value - offset)  # type: ignore[union-attr]
                        if dim != offset_provider.codomain
                        else f"0:{size}"
                        for (dim, offset), size in zip(
                            it.field_domain, field_desc.shape, strict=True
                        )
                    )
                ),
            )
        )
        connectivity_slice = self._construct_local_view(
            MemletExpr(
                dc_node=self.state.add_access(connectivity),
                gt_dtype=node.type,
                subset=sbs.Range.from_string(
                    f"{origin_index.value}, 0:{offset_provider.max_neighbors}"
                ),
            )
        )

        neighbors_temp, _ = self.sdfg.add_temp_transient(
            (offset_provider.max_neighbors,), field_desc.dtype
        )
        neighbors_node = self.state.add_access(neighbors_temp)
        offset_type = gtx_common.Dimension(offset, gtx_common.DimensionKind.LOCAL)
        neighbor_idx = dace_gtir_utils.get_map_variable(offset_type)

        index_connector = "__index"
        output_connector = "__val"
        tasklet_expression = f"{output_connector} = __field[{index_connector}]"
        input_memlets = {
            "__field": self.sdfg.make_array_memlet(field_slice.dc_node.data),
            index_connector: dace.Memlet(data=connectivity_slice.dc_node.data, subset=neighbor_idx),
        }
        input_nodes = {
            field_slice.dc_node.data: field_slice.dc_node,
            connectivity_slice.dc_node.data: connectivity_slice.dc_node,
        }

        if offset_provider.has_skip_values:
            # in case of skip value we can write any dummy value
            skip_value = (
                "math.nan"
                if ti.is_floating_point(node.type.element_type)
                else str(dace.dtypes.max_value(field_desc.dtype))
            )
            tasklet_expression += (
                f" if {index_connector} != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}"
            )

        self._add_mapped_tasklet(
            name=f"{offset}_neighbors",
            map_ranges={neighbor_idx: f"0:{offset_provider.max_neighbors}"},
            code=tasklet_expression,
            inputs=input_memlets,
            input_nodes=input_nodes,
            outputs={
                output_connector: dace.Memlet(data=neighbors_temp, subset=neighbor_idx),
            },
            output_nodes={neighbors_temp: neighbors_node},
            external_edges=True,
        )

        return ValueExpr(
            dc_node=neighbors_node, gt_dtype=itir_ts.ListType(node.type.element_type, offset_type)
        )

    def _visit_map(self, node: gtir.FunCall) -> ValueExpr:
        """
        A map node defines an operation to be mapped on all elements of input arguments.

        The map operation is applied on the local dimension of input fields.
        In the example below, the local dimension consists of a list of neighbor
        values as the first argument, and a list of constant values `1.0`:
        `map_(plus)(neighbors(V2E, it), make_const_list(1.0))`

        The `plus` operation is lowered to a tasklet inside a map that computes
        the domain of the local dimension (in this example, max neighbors in V2E).

        The result is a 1D local field, with same size as the input local dimension.
        In above example, the result would be an array with size V2E.max_neighbors,
        containing the V2E neighbor values incremented by 1.0.
        """
        assert isinstance(node.type, itir_ts.ListType)
        assert isinstance(node.fun, gtir.FunCall)
        assert len(node.fun.args) == 1  # the operation to be mapped on the arguments

        assert isinstance(node.type.element_type, ts.ScalarType)
        dc_dtype = dace_utils.as_dace_type(node.type.element_type)

        input_connectors = [f"__arg{i}" for i in range(len(node.args))]
        output_connector = "__out"

        # Here we build the body of the tasklet
        fun_node = im.call(node.fun.args[0])(*input_connectors)
        fun_python_code = gtir_python_codegen.get_source(fun_node)
        tasklet_expression = f"{output_connector} = {fun_python_code}"

        input_args = [self.visit(arg) for arg in node.args]
        input_connectivity_types: dict[
            gtx_common.Dimension, gtx_common.NeighborConnectivityType
        ] = {}
        for input_arg in input_args:
            assert isinstance(input_arg.gt_dtype, itir_ts.ListType)
            assert input_arg.gt_dtype.offset_type is not None
            offset_type = input_arg.gt_dtype.offset_type
            if offset_type == _CONST_DIM:
                # this input argument is the result of `make_const_list`
                continue
            offset_provider_t = self.subgraph_builder.get_offset_provider_type(offset_type.value)
            assert isinstance(offset_provider_t, gtx_common.NeighborConnectivityType)
            input_connectivity_types[offset_type] = offset_provider_t

        if len(input_connectivity_types) == 0:
            raise ValueError(f"Missing information on local dimension for map node {node}.")

        # GT4Py guarantees that all connectivities used to generate lists of neighbors
        # have the same length, that is the same value of 'max_neighbors'.
        if (
            len(
                set(
                    (conn.has_skip_values, conn.max_neighbors)
                    for conn in input_connectivity_types.values()
                )
            )
            != 1
        ):
            raise ValueError("Unexpected arguments to map expression with different neighborhood.")
        offset_type, offset_provider_type = next(iter(input_connectivity_types.items()))
        local_size = offset_provider_type.max_neighbors
        map_index = dace_gtir_utils.get_map_variable(offset_type)

        # The dataflow we build in this class has some loose connections on input edges.
        # These edges are described as set of nodes, that will have to be connected to
        # external data source nodes passing through the map entry node of the field map.
        # Similarly to `neighbors` expressions, the `map_` input edges terminate on view
        # nodes (see `_construct_local_view` in the for-loop below), because it is simpler
        # than representing map-to-map edges (which require memlets with 2 pass-nodes).
        input_memlets = {}
        input_nodes = {}
        for conn, input_arg in zip(input_connectors, input_args):
            input_node = self._construct_local_view(input_arg).dc_node
            input_desc = input_node.desc(self.sdfg)
            # we assume that there is a single local dimension
            if len(input_desc.shape) != 1:
                raise ValueError(f"More than one local dimension in map expression {node}.")
            input_size = input_desc.shape[0]
            if input_size == 1:
                assert input_arg.gt_dtype.offset_type == _CONST_DIM
                input_memlets[conn] = dace.Memlet(data=input_node.data, subset="0")
            elif input_size == local_size:
                input_memlets[conn] = dace.Memlet(data=input_node.data, subset=map_index)
            else:
                raise ValueError(
                    f"Argument to map node with local size {input_size}, expected {local_size}."
                )
            input_nodes[input_node.data] = input_node

        result, _ = self.sdfg.add_temp_transient((local_size,), dc_dtype)
        result_node = self.state.add_access(result)

        if offset_provider_type.has_skip_values:
            # In case the `map_` input expressions contain skip values, we use
            # the connectivity-based offset provider as mask for map computation.
            connectivity = dace_utils.connectivity_identifier(offset_type.value)
            connectivity_desc = self.sdfg.arrays[connectivity]
            connectivity_desc.transient = False

            origin_map_index = dace_gtir_utils.get_map_variable(offset_provider_type.source_dim)

            connectivity_slice = self._construct_local_view(
                MemletExpr(
                    dc_node=self.state.add_access(connectivity),
                    gt_dtype=itir_ts.ListType(
                        element_type=node.type.element_type, offset_type=offset_type
                    ),
                    subset=sbs.Range.from_string(
                        f"{origin_map_index}, 0:{offset_provider_type.max_neighbors}"
                    ),
                )
            )

            input_memlets["__neighbor_idx"] = dace.Memlet(
                data=connectivity_slice.dc_node.data, subset=map_index
            )
            input_nodes[connectivity_slice.dc_node.data] = connectivity_slice.dc_node

            # in case of skip value we can write any dummy value
            skip_value = (
                "math.nan"
                if ti.is_floating_point(node.type.element_type)
                else str(dace.dtypes.max_value(dc_dtype))
            )
            tasklet_expression += (
                f" if __neighbor_idx != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}"
            )

        self._add_mapped_tasklet(
            name="map",
            map_ranges={map_index: f"0:{local_size}"},
            code=tasklet_expression,
            inputs=input_memlets,
            input_nodes=input_nodes,
            outputs={
                output_connector: dace.Memlet(data=result, subset=map_index),
            },
            output_nodes={result: result_node},
            external_edges=True,
        )

        return ValueExpr(
            dc_node=result_node,
            gt_dtype=itir_ts.ListType(node.type.element_type, offset_type),
        )

    def _make_reduce_with_skip_values(
        self,
        input_expr: ValueExpr | MemletExpr,
        offset_provider_type: gtx_common.NeighborConnectivityType,
        reduce_init: SymbolExpr,
        reduce_identity: SymbolExpr,
        reduce_wcr: str,
        result_node: dace.nodes.AccessNode,
    ) -> None:
        """
        Helper method to lower reduction on a local field containing skip values.

        The reduction is implemented as a nested SDFG containing 2 states. In first
        state, the result (a scalar data node passed as argumet) is initialized.
        In second state, a mapped tasklet uses a write-conflict resolution (wcr)
        memlet to update the result.
        We use the offset provider as a mask to identify skip values: the value
        that is written to the result node is either the input value, when the
        corresponding neighbor index in the connectivity table is valid, or the
        identity value if the neighbor index is missing.
        """
        origin_map_index = dace_gtir_utils.get_map_variable(offset_provider_type.source_dim)

        assert (
            isinstance(input_expr.gt_dtype, itir_ts.ListType)
            and input_expr.gt_dtype.offset_type is not None
        )
        offset_type = input_expr.gt_dtype.offset_type
        connectivity = dace_utils.connectivity_identifier(offset_type.value)
        connectivity_node = self.state.add_access(connectivity)
        connectivity_desc = connectivity_node.desc(self.sdfg)
        connectivity_desc.transient = False

        desc = input_expr.dc_node.desc(self.sdfg)
        if isinstance(input_expr, MemletExpr):
            local_dim_indices = [i for i, size in enumerate(input_expr.subset.size()) if size != 1]
        else:
            local_dim_indices = list(range(len(desc.shape)))

        if len(local_dim_indices) != 1:
            raise NotImplementedError(
                f"Found {len(local_dim_indices)} local dimensions in reduce expression, expected one."
            )
        local_dim_index = local_dim_indices[0]
        assert desc.shape[local_dim_index] == offset_provider_type.max_neighbors

        # we lower the reduction map with WCR out memlet in a nested SDFG
        nsdfg = dace.SDFG(name=self.unique_nsdfg_name("reduce_with_skip_values"))
        nsdfg.add_array(
            "values",
            (desc.shape[local_dim_index],),
            desc.dtype,
            strides=(desc.strides[local_dim_index],),
        )
        nsdfg.add_array(
            "neighbor_indices",
            (connectivity_desc.shape[1],),
            connectivity_desc.dtype,
            strides=(connectivity_desc.strides[1],),
        )
        nsdfg.add_scalar("acc", desc.dtype)
        st_init = nsdfg.add_state(f"{nsdfg.label}_init")
        st_init.add_edge(
            st_init.add_tasklet(
                "init_acc",
                {},
                {"__val"},
                f"__val = {reduce_init.dc_dtype}({reduce_init.value})",
            ),
            "__val",
            st_init.add_access("acc"),
            None,
            dace.Memlet(data="acc", subset="0"),
        )
        st_reduce = nsdfg.add_state_after(st_init, f"{nsdfg.label}_reduce")
        # Fill skip values in local dimension with the reduce identity value
        skip_value = f"{reduce_identity.dc_dtype}({reduce_identity.value})"
        # Since this map operates on a pure local dimension, we explicitly set sequential
        # schedule and we set the flag 'wcr_nonatomic=True' on the write memlet.
        # TODO(phimuell): decide if auto-optimizer should reset `wcr_nonatomic` properties, as DaCe does.
        st_reduce.add_mapped_tasklet(
            name="reduce_with_skip_values",
            map_ranges={"i": f"0:{offset_provider_type.max_neighbors}"},
            inputs={
                "__val": dace.Memlet(data="values", subset="i"),
                "__neighbor_idx": dace.Memlet(data="neighbor_indices", subset="i"),
            },
            code=f"__out = __val if __neighbor_idx != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}",
            outputs={
                "__out": dace.Memlet(data="acc", subset="0", wcr=reduce_wcr, wcr_nonatomic=True),
            },
            external_edges=True,
            schedule=dace.dtypes.ScheduleType.Sequential,
        )

        nsdfg_node = self.state.add_nested_sdfg(
            nsdfg, self.sdfg, inputs={"values", "neighbor_indices"}, outputs={"acc"}
        )

        if isinstance(input_expr, MemletExpr):
            self._add_input_data_edge(input_expr.dc_node, input_expr.subset, nsdfg_node, "values")
        else:
            self.state.add_edge(
                input_expr.dc_node,
                None,
                nsdfg_node,
                "values",
                self.sdfg.make_array_memlet(input_expr.dc_node.data),
            )
        self._add_input_data_edge(
            connectivity_node,
            sbs.Range.from_string(f"{origin_map_index}, 0:{offset_provider_type.max_neighbors}"),
            nsdfg_node,
            "neighbor_indices",
        )
        self.state.add_edge(
            nsdfg_node,
            "acc",
            result_node,
            None,
            dace.Memlet(data=result_node.data, subset="0"),
        )

    def _visit_reduce(self, node: gtir.FunCall) -> ValueExpr:
        assert isinstance(node.type, ts.ScalarType)
        op_name, reduce_init, reduce_identity = get_reduce_params(node)
        reduce_wcr = "lambda x, y: " + gtir_python_codegen.format_builtin(op_name, "x", "y")

        result = self.sdfg.temp_data_name()
        self.sdfg.add_scalar(result, reduce_identity.dc_dtype, transient=True)
        result_node = self.state.add_access(result)

        input_expr = self.visit(node.args[0])
        assert isinstance(input_expr, (MemletExpr, ValueExpr))
        assert (
            isinstance(input_expr.gt_dtype, itir_ts.ListType)
            and input_expr.gt_dtype.offset_type is not None
        )
        offset_type = input_expr.gt_dtype.offset_type
        offset_provider_type = self.subgraph_builder.get_offset_provider_type(offset_type.value)
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)

        if offset_provider_type.has_skip_values:
            self._make_reduce_with_skip_values(
                input_expr,
                offset_provider_type,
                reduce_init,
                reduce_identity,
                reduce_wcr,
                result_node,
            )

        else:
            reduce_node = self.state.add_reduce(reduce_wcr, axes=None, identity=reduce_init.value)
            if isinstance(input_expr, MemletExpr):
                self._add_input_data_edge(input_expr.dc_node, input_expr.subset, reduce_node)
            else:
                self.state.add_nedge(
                    input_expr.dc_node,
                    reduce_node,
                    self.sdfg.make_array_memlet(input_expr.dc_node.data),
                )
            self.state.add_nedge(reduce_node, result_node, dace.Memlet(data=result, subset="0"))

        return ValueExpr(result_node, node.type)

    def _split_shift_args(
        self, args: list[gtir.Expr]
    ) -> tuple[tuple[gtir.Expr, gtir.Expr], Optional[list[gtir.Expr]]]:
        """
        Splits the arguments to `shift` builtin function as pairs, each pair containing
        the offset provider and the offset expression in one dimension.
        """
        nargs = len(args)
        assert nargs >= 2 and nargs % 2 == 0
        return (args[-2], args[-1]), args[: nargs - 2] if nargs > 2 else None

    def _visit_shift_multidim(
        self, iterator: gtir.Expr, shift_args: list[gtir.Expr]
    ) -> tuple[gtir.Expr, gtir.Expr, IteratorExpr]:
        """Transforms a multi-dimensional shift into recursive shift calls, each in a single dimension."""
        (offset_provider_arg, offset_value_arg), tail = self._split_shift_args(shift_args)
        if tail:
            node = gtir.FunCall(
                fun=gtir.FunCall(fun=gtir.SymRef(id="shift"), args=tail),
                args=[iterator],
            )
            it = self.visit(node)
        else:
            it = self.visit(iterator)

        assert isinstance(it, IteratorExpr)
        return offset_provider_arg, offset_value_arg, it

    def _make_cartesian_shift(
        self, it: IteratorExpr, offset_dim: gtx_common.Dimension, offset_expr: DataExpr
    ) -> IteratorExpr:
        """Implements cartesian shift along one dimension."""
        assert any(dim == offset_dim for dim, _ in it.field_domain)
        new_index: SymbolExpr | ValueExpr
        index_expr = it.indices[offset_dim]
        if isinstance(index_expr, SymbolExpr) and isinstance(offset_expr, SymbolExpr):
            # purely symbolic expression which can be interpreted at compile time
            new_index = SymbolExpr(
                index_expr.value + offset_expr.value,
                index_expr.dc_dtype,
            )
        else:
            # the offset needs to be calculated by means of a tasklet (i.e. dynamic offset)
            new_index_connector = "shifted_index"
            if isinstance(index_expr, SymbolExpr):
                dynamic_offset_tasklet = self._add_tasklet(
                    "dynamic_offset",
                    {"offset"},
                    {new_index_connector},
                    f"{new_index_connector} = {index_expr.value} + offset",
                )
            elif isinstance(offset_expr, SymbolExpr):
                dynamic_offset_tasklet = self._add_tasklet(
                    "dynamic_offset",
                    {"index"},
                    {new_index_connector},
                    f"{new_index_connector} = index + {offset_expr}",
                )
            else:
                dynamic_offset_tasklet = self._add_tasklet(
                    "dynamic_offset",
                    {"index", "offset"},
                    {new_index_connector},
                    f"{new_index_connector} = index + offset",
                )
            for input_expr, input_connector in [(index_expr, "index"), (offset_expr, "offset")]:
                if isinstance(input_expr, MemletExpr):
                    self._add_input_data_edge(
                        input_expr.dc_node,
                        input_expr.subset,
                        dynamic_offset_tasklet,
                        input_connector,
                    )
                elif isinstance(input_expr, ValueExpr):
                    self._add_edge(
                        input_expr.dc_node,
                        None,
                        dynamic_offset_tasklet,
                        input_connector,
                        dace.Memlet(data=input_expr.dc_node.data, subset="0"),
                    )

            if isinstance(index_expr, SymbolExpr):
                dc_dtype = index_expr.dc_dtype
            else:
                dc_dtype = index_expr.dc_node.desc(self.sdfg).dtype

            new_index = self._construct_tasklet_result(
                dc_dtype, dynamic_offset_tasklet, new_index_connector
            )

        # a new iterator with a shifted index along one dimension
        shifted_indices = {
            dim: (new_index if dim == offset_dim else index) for dim, index in it.indices.items()
        }
        return IteratorExpr(it.field, it.gt_dtype, it.field_domain, shifted_indices)

    def _make_dynamic_neighbor_offset(
        self,
        offset_expr: MemletExpr | ValueExpr,
        offset_table_node: dace.nodes.AccessNode,
        origin_index: SymbolExpr,
    ) -> ValueExpr:
        """
        Implements access to neighbor connectivity table by means of a tasklet node.

        It requires a dynamic offset value, either obtained from a field/scalar argument (`MemletExpr`)
        or computed by another tasklet (`DataExpr`).
        """
        new_index_connector = "neighbor_index"
        tasklet_node = self._add_tasklet(
            "dynamic_neighbor_offset",
            {"table", "offset"},
            {new_index_connector},
            f"{new_index_connector} = table[{origin_index.value}, offset]",
        )
        self._add_input_data_edge(
            offset_table_node,
            sbs.Range.from_array(offset_table_node.desc(self.sdfg)),
            tasklet_node,
            "table",
        )
        if isinstance(offset_expr, MemletExpr):
            self._add_input_data_edge(
                offset_expr.dc_node,
                offset_expr.subset,
                tasklet_node,
                "offset",
            )
        else:
            self._add_edge(
                offset_expr.dc_node,
                None,
                tasklet_node,
                "offset",
                dace.Memlet(data=offset_expr.dc_node.data, subset="0"),
            )

        dc_dtype = offset_table_node.desc(self.sdfg).dtype
        return self._construct_tasklet_result(dc_dtype, tasklet_node, new_index_connector)

    def _make_unstructured_shift(
        self,
        it: IteratorExpr,
        connectivity: gtx_common.NeighborConnectivityType,
        offset_table_node: dace.nodes.AccessNode,
        offset_expr: DataExpr,
    ) -> IteratorExpr:
        """Implements shift in unstructured domain by means of a neighbor table."""
        assert any(dim == connectivity.codomain for dim, _ in it.field_domain)
        neighbor_dim = connectivity.codomain
        assert neighbor_dim not in it.indices

        origin_dim = connectivity.source_dim
        assert origin_dim in it.indices
        origin_index = it.indices[origin_dim]
        assert isinstance(origin_index, SymbolExpr)

        shifted_indices = {dim: idx for dim, idx in it.indices.items() if dim != origin_dim}
        if isinstance(offset_expr, SymbolExpr):
            # use memlet to retrieve the neighbor index
            shifted_indices[neighbor_dim] = MemletExpr(
                dc_node=offset_table_node,
                gt_dtype=it.gt_dtype,
                subset=sbs.Indices([origin_index.value, offset_expr.value]),
            )
        else:
            # dynamic offset: we cannot use a memlet to retrieve the offset value, use a tasklet node
            shifted_indices[neighbor_dim] = self._make_dynamic_neighbor_offset(
                offset_expr, offset_table_node, origin_index
            )

        return IteratorExpr(it.field, it.gt_dtype, it.field_domain, shifted_indices)

    def _visit_shift(self, node: gtir.FunCall) -> IteratorExpr:
        # convert builtin-index type to dace type
        IndexDType: Final = dace_utils.as_dace_type(
            ts.ScalarType(kind=getattr(ts.ScalarKind, gtir.INTEGER_INDEX_BUILTIN.upper()))
        )

        assert isinstance(node.fun, gtir.FunCall)
        # the iterator to be shifted is the node argument, while the shift arguments
        # are provided by the nested function call; the shift arguments consist of
        # the offset provider and the offset value in each dimension to be shifted
        offset_provider_arg, offset_value_arg, it = self._visit_shift_multidim(
            node.args[0], node.fun.args
        )

        # first argument of the shift node is the offset provider
        assert isinstance(offset_provider_arg, gtir.OffsetLiteral)
        offset = offset_provider_arg.value
        assert isinstance(offset, str)
        offset_provider_type = self.subgraph_builder.get_offset_provider_type(offset)
        # second argument should be the offset value, which could be a symbolic expression or a dynamic offset
        offset_expr = (
            SymbolExpr(offset_value_arg.value, IndexDType)
            if isinstance(offset_value_arg, gtir.OffsetLiteral)
            else self.visit(offset_value_arg)
        )

        if isinstance(offset_provider_type, gtx_common.Dimension):
            return self._make_cartesian_shift(it, offset_provider_type, offset_expr)
        else:
            # initially, the storage for the connectivity tables is created as transient;
            # when the tables are used, the storage is changed to non-transient,
            # so the corresponding arrays are supposed to be allocated by the SDFG caller
            offset_table = dace_utils.connectivity_identifier(offset)
            self.sdfg.arrays[offset_table].transient = False
            offset_table_node = self.state.add_access(offset_table)

            return self._make_unstructured_shift(
                it, offset_provider_type, offset_table_node, offset_expr
            )

    def _visit_generic_builtin(self, node: gtir.FunCall) -> ValueExpr:
        """
        Generic handler called by `visit_FunCall()` when it encounters
        a builtin function that does not match any other specific handler.
        """
        node_internals = []
        node_connections: dict[str, MemletExpr | ValueExpr] = {}
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, MemletExpr | ValueExpr):
                # the argument value is the result of a tasklet node or direct field access
                connector = f"__arg{i}"
                node_connections[connector] = arg_expr
                node_internals.append(connector)
            else:
                assert isinstance(arg_expr, SymbolExpr)
                # use the argument value without adding any connector
                node_internals.append(arg_expr.value)

        assert isinstance(node.fun, gtir.SymRef)
        builtin_name = str(node.fun.id)
        # use tasklet connectors as expression arguments
        code = gtir_python_codegen.format_builtin(builtin_name, *node_internals)

        out_connector = "result"
        tasklet_node = self._add_tasklet(
            builtin_name,
            set(node_connections.keys()),
            {out_connector},
            "{} = {}".format(out_connector, code),
        )

        for connector, arg_expr in node_connections.items():
            if isinstance(arg_expr, ValueExpr):
                self._add_edge(
                    arg_expr.dc_node,
                    None,
                    tasklet_node,
                    connector,
                    dace.Memlet(data=arg_expr.dc_node.data, subset="0"),
                )
            else:
                self._add_input_data_edge(
                    arg_expr.dc_node,
                    arg_expr.subset,
                    tasklet_node,
                    connector,
                )

        if isinstance(node.type, itir_ts.ListType):
            # The only builtin function (so far) handled here that returns a list
            # is 'make_const_list'. There are other builtin functions (map_, neighbors)
            # that return a list but they are handled in specialized visit methods.
            # This method (the generic visitor for builtin functions) always returns
            # a single value. This is also the case of 'make_const_list' expression:
            # it simply broadcasts a scalar on the local domain of another expression,
            # for example 'map_(plus)(neighbors(V2Eₒ, it), make_const_list(1.0))'.
            # Therefore we handle `ListType` as a single-element array with shape (1,)
            # that will be accessed in a map expression on a local domain.
            assert isinstance(node.type.element_type, ts.ScalarType)
            dc_dtype = dace_utils.as_dace_type(node.type.element_type)
            # In order to ease the lowring of the parent expression on local dimension,
            # we represent the scalar value as a single-element 1D array.
            use_array = True
        else:
            assert isinstance(node.type, ts.ScalarType)
            dc_dtype = dace_utils.as_dace_type(node.type)
            use_array = False

        return self._construct_tasklet_result(dc_dtype, tasklet_node, "result", use_array=use_array)

    def visit_FunCall(self, node: gtir.FunCall) -> IteratorExpr | DataExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node, "neighbors"):
            return self._visit_neighbors(node)

        elif cpm.is_applied_map(node):
            return self._visit_map(node)

        elif cpm.is_applied_reduce(node):
            return self._visit_reduce(node)

        elif cpm.is_applied_shift(node):
            return self._visit_shift(node)

        elif isinstance(node.fun, gtir.SymRef):
            return self._visit_generic_builtin(node)

        else:
            raise NotImplementedError(f"Invalid 'FunCall' node: {node}.")

    def visit_Lambda(
        self, node: gtir.Lambda, args: list[IteratorExpr | MemletExpr | SymbolExpr]
    ) -> tuple[list[DataflowInputEdge], DataflowOutputEdge]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr: DataExpr = self.visit(node.expr)
        if isinstance(output_expr, ValueExpr):
            return self.input_edges, DataflowOutputEdge(self.state, output_expr)

        if isinstance(output_expr, MemletExpr):
            # special case where the field operator is simply copying data from source to destination node
            output_dtype = output_expr.dc_node.desc(self.sdfg).dtype
            tasklet_node = self._add_tasklet("copy", {"__inp"}, {"__out"}, "__out = __inp")
            self._add_input_data_edge(
                output_expr.dc_node,
                output_expr.subset,
                tasklet_node,
                "__inp",
            )
        else:
            assert isinstance(output_expr, SymbolExpr)
            # even simpler case, where a constant value is written to destination node
            output_dtype = output_expr.dc_dtype
            tasklet_node = self._add_tasklet("write", {}, {"__out"}, f"__out = {output_expr.value}")

        output_expr = self._construct_tasklet_result(output_dtype, tasklet_node, "__out")
        return self.input_edges, DataflowOutputEdge(self.state, output_expr)

    def visit_Literal(self, node: gtir.Literal) -> SymbolExpr:
        dc_dtype = dace_utils.as_dace_type(node.type)
        return SymbolExpr(node.value, dc_dtype)

    def visit_SymRef(self, node: gtir.SymRef) -> IteratorExpr | MemletExpr | SymbolExpr:
        param = str(node.id)
        if param in self.symbol_map:
            return self.symbol_map[param]
        # if not in the lambda symbol map, this must be a symref to a builtin function
        assert param in gtir_python_codegen.MATH_BUILTINS_MAPPING
        return SymbolExpr(param, dace.string)
