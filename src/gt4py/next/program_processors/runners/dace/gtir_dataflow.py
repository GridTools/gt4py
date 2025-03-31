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
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    Union,
)

import dace
from dace import subsets as dace_subsets

from gt4py import eve
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import builtins as gtir_builtins, ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.program_processors.runners.dace import (
    gtir_python_codegen,
    gtir_sdfg,
    gtir_sdfg_utils,
    utils as gtx_dace_utils,
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
    gt_dtype: ts.ListType | ts.ScalarType


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
    gt_dtype: ts.ListType | ts.ScalarType
    subset: dace_subsets.Range


@dataclasses.dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymbolicType
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
    gt_dtype: ts.ListType | ts.ScalarType
    field_domain: list[tuple[gtx_common.Dimension, dace.symbolic.SymbolicType]]
    indices: dict[gtx_common.Dimension, DataExpr]

    def get_field_type(self) -> ts.FieldType:
        return ts.FieldType([dim for dim, _ in self.field_domain], self.gt_dtype)

    def get_memlet_subset(self, sdfg: dace.SDFG) -> dace_subsets.Range:
        if len(self.field_domain) == 0:  # zero-dimensional field
            return dace_subsets.Range.from_string("0")

        if not all(isinstance(self.indices[dim], SymbolExpr) for dim, _ in self.field_domain):
            raise ValueError(f"Cannot deref iterator {self}.")

        field_desc = self.field.desc(sdfg)
        if isinstance(self.gt_dtype, ts.ListType):
            assert len(field_desc.shape) == len(self.field_domain) + 1
            assert self.gt_dtype.offset_type is not None
            field_domain = [*self.field_domain, (self.gt_dtype.offset_type, 0)]
        else:
            assert len(field_desc.shape) == len(self.field_domain)
            field_domain = self.field_domain

        return dace_subsets.Range.from_string(
            ",".join(
                str(self.indices[dim].value - offset)  # type: ignore[union-attr]
                if dim in self.indices
                else f"0:{size}"
                for (dim, offset), size in zip(field_domain, field_desc.shape, strict=True)
            )
        )


class DataflowInputEdge(Protocol):
    """
    This protocol describes how to concretize a data edge to read data from a source node
    into the dataflow.

    It provides the `connect` method to setup an input edge from an external data source.
    The most common case is that the dataflow represents a stencil, which is instantied
    inside a map scope and whose inputs and outputs are connected to external data nodes
    by means of memlets that traverse the map entry and exit nodes.
    The dataflow can also be instatiated without a map, in which case the `map_entry`
    argument is set to `None`.
    """

    @abc.abstractmethod
    def connect(self, map_entry: Optional[dace.nodes.MapEntry]) -> None: ...


@dataclasses.dataclass(frozen=True)
class MemletInputEdge(DataflowInputEdge):
    """
    Allows to setup an input memlet through a map entry node.

    The edge source has to be a data access node, while the destination node can either
    be a tasklet, in which case the connector name is also required, or an access node.
    """

    state: dace.SDFGState
    source: dace.nodes.AccessNode
    subset: dace_subsets.Range
    dest: dace.nodes.AccessNode | dace.nodes.Tasklet
    dest_conn: Optional[str]

    def connect(self, map_entry: Optional[dace.nodes.MapEntry]) -> None:
        memlet = dace.Memlet(data=self.source.data, subset=self.subset)
        if map_entry is None:
            self.state.add_edge(self.source, None, self.dest, self.dest_conn, memlet)
        else:
            self.state.add_memlet_path(
                self.source,
                map_entry,
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

    def connect(self, map_entry: Optional[dace.nodes.MapEntry]) -> None:
        if map_entry is None:
            # outside of a map scope it is possible to instantiate a tasklet node
            # without input connectors
            return
        self.state.add_nedge(map_entry, self.node, dace.Memlet())


@dataclasses.dataclass(frozen=True)
class DataflowOutputEdge:
    """
    Allows to setup an output memlet through a map exit node.

    The result of a dataflow subgraph needs to be written to an external data node.
    The most common case is that the dataflow represents a stencil and the dataflow
    is computed over a field domain, therefore the dataflow is instatiated inside
    a map scope. The `connect` method creates a memlet that writes the dataflow
    result to the external array passing through the `map_exit` node.
    The dataflow can also be instatiated without a map, in which case the `map_exit`
    argument is set to `None`.
    """

    state: dace.SDFGState
    result: ValueExpr

    def connect(
        self,
        map_exit: Optional[dace.nodes.MapExit],
        dest: dace.nodes.AccessNode,
        subset: dace_subsets.Range,
    ) -> None:
        write_edge = self.state.in_edges(self.result.dc_node)[0]
        write_size = write_edge.data.dst_subset.num_elements()
        # check the kind of node which writes the result
        if isinstance(write_edge.src, dace.nodes.Tasklet):
            # The temporary data written by a tasklet can be safely deleted
            assert write_size.is_constant()
            remove_last_node = True
        elif isinstance(write_edge.src, dace.nodes.NestedSDFG):
            if write_size.is_constant():
                # Temporary data with compile-time size is allocated on the stack
                # and therefore is safe to keep. We decide to keep it as a workaround
                # for a dace issue with memlet propagation in combination with
                # nested SDFGs containing conditional blocks. The output memlet
                # of such blocks will be marked as dynamic because dace is not able
                # to detect the exact size of a conditional branch dataflow, even
                # in case of if-else expressions with exact same output data.
                remove_last_node = False
            else:
                # In case the output data has runtime size it is necessary to remove
                # it in order to avoid dynamic memory allocation inside a parallel
                # map scope. Otherwise, the memory allocation will for sure lead
                # to performance degradation, and eventually illegal memory issues
                # when the gpu runs out of local memory.
                remove_last_node = True
        else:
            remove_last_node = False

        if remove_last_node:
            last_node = write_edge.src
            last_node_connector = write_edge.src_conn
            self.state.remove_node(self.result.dc_node)
        else:
            last_node = self.result.dc_node
            last_node_connector = None

        if map_exit is None:
            self.state.add_edge(
                last_node,
                last_node_connector,
                dest,
                None,
                dace.Memlet(data=dest.data, subset=subset),
            )
        else:
            self.state.add_memlet_path(
                last_node,
                map_exit,
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
    dc_dtype = gtx_dace_utils.as_dace_type(node.type)

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


def get_tuple_type(
    data: tuple[IteratorExpr | MemletExpr | ValueExpr | tuple[Any, ...], ...],
) -> ts.TupleType:
    """
    Compute the `ts.TupleType` corresponding to the tuple structure of input data expressions.
    """
    data_types: list[ts.DataType] = []
    for dataitem in data:
        if isinstance(dataitem, tuple):
            data_types.append(get_tuple_type(dataitem))
        elif isinstance(dataitem, IteratorExpr):
            data_types.append(dataitem.get_field_type())
        elif isinstance(dataitem, MemletExpr):
            data_types.append(dataitem.gt_dtype)
        else:
            data_types.append(dataitem.gt_dtype)
    return ts.TupleType(data_types)


@dataclasses.dataclass(frozen=True)
class LambdaToDataflow(eve.NodeVisitor):
    """
    Visitor class to translate a `Lambda` expression to a dataflow graph.

    This visitor should be applied by calling `apply()` method on a `Lambda` IR.
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
    input_edges: list[DataflowInputEdge] = dataclasses.field(default_factory=lambda: [])
    symbol_map: dict[
        str,
        IteratorExpr | DataExpr | tuple[IteratorExpr | DataExpr | tuple[Any, ...], ...],
    ] = dataclasses.field(default_factory=dict)

    def _add_input_data_edge(
        self,
        src: dace.nodes.AccessNode,
        src_subset: dace_subsets.Range,
        dst_node: dace.nodes.Node,
        dst_conn: Optional[str] = None,
        src_offset: Optional[list[dace.symbolic.SymExpr]] = None,
    ) -> None:
        input_subset = (
            src_subset
            if src_offset is None
            else dace_subsets.Range(
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
        inputs: Dict[str, dace.Memlet],
        code: str,
        outputs: Dict[str, dace.Memlet],
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
                f"view_{field.dc_node.data}",
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
        data_type = gtx_dace_utils.as_itir_type(dc_dtype)
        if use_array:
            # In some cases, such as result data with list-type annotation, we want
            # that output data is represented as an array (single-element 1D array)
            # in order to allow for composition of array shape in external memlets.
            temp_name, _ = self.subgraph_builder.add_temp_array(self.sdfg, (1,), dc_dtype)
        else:
            temp_name, _ = self.subgraph_builder.add_temp_scalar(self.sdfg, dc_dtype)

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
                ts.ListType(element_type=data_type, offset_type=_CONST_DIM)
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

        if isinstance(node.type, ts.TupleType):
            raise NotImplementedError("Tuple deref not supported.")

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

        # handle default case below: deref a field with one or more dimensions

        # when the indices are all dace symbolic expressions, the deref is lowered
        # to a memlet, where the index is the memlet subset
        if all(isinstance(index, SymbolExpr) for index in arg_expr.indices.values()):
            # when all indices are symbolic expressions, we can perform direct field access through a memlet
            field_subset = arg_expr.get_memlet_subset(self.sdfg)
            return MemletExpr(arg_expr.field, arg_expr.gt_dtype, field_subset)

        # when any of the indices is a runtime value (either a dynamic cartesian
        # offset or a connectivity offset), the deref is lowered to a tasklet
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
            "deref",
            {"field"} | set(index_connectors),
            {"val"},
            code=f"val = field[{index_internals}]",
        )
        # add new termination point for the field parameter
        self._add_input_data_edge(
            arg_expr.field,
            dace_subsets.Range.from_array(field_desc),
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

    def _visit_if_branch_arg(
        self,
        if_sdfg: dace.SDFG,
        if_branch_state: dace.SDFGState,
        param_name: str,
        arg: IteratorExpr | DataExpr,
        deref_on_input_memlet: bool,
        if_sdfg_input_memlets: dict[str, MemletExpr | ValueExpr],
    ) -> IteratorExpr | ValueExpr:
        """
        Helper method to be called by `_visit_if_branch()` to visit the input arguments.

        Args:
            if_sdfg: The nested SDFG where the if expression is lowered.
            if_branch_state: The state inside the nested SDFG where the if branch is lowered.
            param_name: The parameter name of the input argument.
            arg: The input argument expression.
            deref_on_input_memlet: When True, the given iterator argument can be dereferenced on the input memlet.
            if_sdfg_input_memlets: The memlets that provide input data to the nested SDFG, will be update inside this function.
        """
        use_full_shape = False
        if isinstance(arg, (MemletExpr, ValueExpr)):
            arg_desc = arg.dc_node.desc(self.sdfg)
            arg_expr = arg
        elif isinstance(arg, IteratorExpr):
            arg_desc = arg.field.desc(self.sdfg)
            if deref_on_input_memlet:
                # If the iterator is just dereferenced inside the branch state,
                # we can access the array outside the nested SDFG and pass the
                # local data. This approach makes the data dependencies of nested
                # structures more explicit and thus makes it easier for MapFusion
                # to correctly infer the data dependencies.
                memlet_subset = arg.get_memlet_subset(self.sdfg)
                arg_expr = MemletExpr(arg.field, arg.gt_dtype, memlet_subset)
            else:
                # In order to shift the iterator inside the branch dataflow,
                # we have to pass the full array shape.
                arg_expr = MemletExpr(
                    arg.field, arg.gt_dtype, dace_subsets.Range.from_array(arg_desc)
                )
                use_full_shape = True
        else:
            raise TypeError(f"Unexpected {arg} as input argument.")

        if use_full_shape:
            inner_desc = arg_desc.clone()
            inner_desc.transient = False
        elif isinstance(arg.gt_dtype, ts.ScalarType):
            inner_desc = dace.data.Scalar(arg_desc.dtype)
        else:
            # for list of values, we retrieve the local size from the corresponding offset
            assert arg.gt_dtype.offset_type is not None
            offset_provider_type = self.subgraph_builder.get_offset_provider_type(
                arg.gt_dtype.offset_type.value
            )
            assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
            inner_desc = dace.data.Array(arg_desc.dtype, [offset_provider_type.max_neighbors])

        if param_name in if_sdfg.arrays:
            # the data desciptor was added by the visitor of the other branch expression
            assert if_sdfg.data(param_name) == inner_desc
        else:
            if_sdfg.add_datadesc(param_name, inner_desc)
            if_sdfg_input_memlets[param_name] = arg_expr

        inner_node = if_branch_state.add_access(param_name)
        if isinstance(arg, IteratorExpr) and use_full_shape:
            return IteratorExpr(inner_node, arg.gt_dtype, arg.field_domain, arg.indices)
        else:
            return ValueExpr(inner_node, arg.gt_dtype)

    def _visit_if_branch(
        self,
        if_sdfg: dace.SDFG,
        if_branch_state: dace.SDFGState,
        expr: gtir.Expr,
        if_sdfg_input_memlets: dict[str, MemletExpr | ValueExpr],
        direct_deref_iterators: Iterable[str],
    ) -> tuple[
        list[DataflowInputEdge],
        tuple[DataflowOutputEdge | tuple[Any, ...], ...],
    ]:
        """
        Helper method to visit an if-branch expression and lower it to a dataflow inside the given nested SDFG and state.

        This function is called by `_visit_if()` for each if-branch.

        Args:
            if_sdfg: The nested SDFG where the if expression is lowered.
            if_branch_state: The state inside the nested SDFG where the if branch is lowered.
            expr: The if branch expression to lower.
            if_sdfg_input_memlets: The memlets that provide input data to the nested SDFG, will be update inside this function.
            direct_deref_iterators: Fields that are accessed with direct iterator deref, without any shift.

        Returns:
            A tuple containing:
                - the list of input edges for the parent dataflow
                - the tree representation of output data, in the form of a tuple of data edges.
        """
        assert if_branch_state in if_sdfg.states()

        lambda_args = []
        lambda_params = []
        for pname in symbol_ref_utils.collect_symbol_refs(expr, self.symbol_map.keys()):
            arg = self.symbol_map[pname]
            if isinstance(arg, tuple):
                ptype = get_tuple_type(arg)  # type: ignore[arg-type]
                psymbol = im.sym(pname, ptype)
                psymbol_tree = gtir_sdfg_utils.make_symbol_tree(pname, ptype)
                deref_on_input_memlet = pname in direct_deref_iterators
                inner_arg = gtx_utils.tree_map(
                    lambda tsym,
                    targ,
                    deref_on_input_memlet=deref_on_input_memlet: self._visit_if_branch_arg(
                        if_sdfg,
                        if_branch_state,
                        tsym.id,
                        targ,
                        deref_on_input_memlet,
                        if_sdfg_input_memlets,
                    )
                )(psymbol_tree, arg)
            else:
                psymbol = im.sym(pname, arg.gt_dtype)  # type: ignore[union-attr]
                deref_on_input_memlet = pname in direct_deref_iterators
                inner_arg = self._visit_if_branch_arg(
                    if_sdfg,
                    if_branch_state,
                    pname,
                    arg,
                    deref_on_input_memlet,
                    if_sdfg_input_memlets,
                )
            lambda_args.append(inner_arg)
            lambda_params.append(psymbol)

        # visit each branch of the if-statement as if it was a Lambda node
        lambda_node = gtir.Lambda(params=lambda_params, expr=expr)
        input_edges, output_tree = translate_lambda_to_dataflow(
            if_sdfg, if_branch_state, self.subgraph_builder, lambda_node, lambda_args
        )

        for data_node in if_branch_state.data_nodes():
            # In case of tuple arguments, isolated access nodes might be left in the state,
            # because not all tuple fields are necessarily used inside the lambda scope
            if if_branch_state.degree(data_node) == 0:
                assert not data_node.desc(if_sdfg).transient
                if_branch_state.remove_node(data_node)

        return input_edges, output_tree

    def _visit_if_branch_result(
        self, sdfg: dace.SDFG, state: dace.SDFGState, edge: DataflowOutputEdge, sym: gtir.Sym
    ) -> ValueExpr:
        """
        Helper function to be called by `_visit_if` to create an output connector
        on the nested SDFG that will write the result to the parent SDFG.
        The result data inside the nested SDFG must have the same name as the connector.
        """
        output_data = str(sym.id)
        if output_data in sdfg.arrays:
            output_desc = sdfg.data(output_data)
            assert not output_desc.transient
        else:
            # If the result is currently written to a transient node, inside the nested SDFG,
            # we need to allocate a non-transient data node.
            result_desc = edge.result.dc_node.desc(sdfg)
            output_desc = result_desc.clone()
            output_desc.transient = False
            output_data = sdfg.add_datadesc(output_data, output_desc, find_new_name=True)
        output_node = state.add_access(output_data)
        state.add_nedge(
            edge.result.dc_node,
            output_node,
            dace.Memlet.from_array(output_data, output_desc),
        )
        return ValueExpr(output_node, edge.result.gt_dtype)

    def _visit_if(self, node: gtir.FunCall) -> ValueExpr | tuple[ValueExpr | tuple[Any, ...], ...]:
        """
        Lowers an if-expression with exclusive branch execution into a nested SDFG,
        in which each branch is lowered into a dataflow in a separate state and
        the if-condition is represented as the inter-state edge condition.
        """

        def write_output_of_nested_sdfg_to_temporary(inner_value: ValueExpr) -> ValueExpr:
            # Each output connector of the nested SDFG writes to a transient node in the parent SDFG
            inner_data = inner_value.dc_node.data
            inner_desc = inner_value.dc_node.desc(nsdfg)
            assert not inner_desc.transient
            output, output_desc = self.subgraph_builder.add_temp_array_like(self.sdfg, inner_desc)
            output_node = self.state.add_access(output)
            self.state.add_edge(
                nsdfg_node,
                inner_data,
                output_node,
                None,
                dace.Memlet.from_array(output, output_desc),
            )
            return ValueExpr(output_node, inner_value.gt_dtype)

        assert len(node.args) == 3

        # evaluate the if-condition that will write to a boolean scalar node
        condition_value = self.visit(node.args[0])
        assert (
            (
                isinstance(condition_value.gt_dtype, ts.ScalarType)
                and condition_value.gt_dtype.kind == ts.ScalarKind.BOOL
            )
            if isinstance(condition_value, (MemletExpr, ValueExpr))
            else (condition_value.dc_dtype == dace.dtypes.bool_)
        )

        nsdfg = dace.SDFG(self.unique_nsdfg_name(prefix="if_stmt"))
        nsdfg.debuginfo = gtir_sdfg_utils.debug_info(node, default=self.sdfg.debuginfo)

        # create states inside the nested SDFG for the if-branches
        if_region = dace.sdfg.state.ConditionalBlock("if")
        nsdfg.add_node(if_region)
        entry_state = nsdfg.add_state("entry", is_start_block=True)
        nsdfg.add_edge(entry_state, if_region, dace.InterstateEdge())

        then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=nsdfg)
        tstate = then_body.add_state("true_branch", is_start_block=True)
        if_region.add_branch(dace.sdfg.state.CodeBlock("__cond"), then_body)

        else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=nsdfg)
        fstate = else_body.add_state("false_branch", is_start_block=True)
        if_region.add_branch(dace.sdfg.state.CodeBlock("not (__cond)"), else_body)

        input_memlets: dict[str, MemletExpr | ValueExpr] = {}
        nsdfg_symbols_mapping: Optional[dict[str, dace.symbol]] = None

        # define scalar or symbol for the condition value inside the nested SDFG
        if isinstance(condition_value, SymbolExpr):
            nsdfg.add_symbol("__cond", dace.dtypes.bool)
        else:
            nsdfg.add_scalar("__cond", dace.dtypes.bool)
            input_memlets["__cond"] = condition_value

        # Collect all field iterators that are shifted inside any of the then/else
        # branch expressions. Iterator shift expressions require the field argument
        # as iterator, therefore the corresponding array has to be passed with full
        # shape into the nested SDFG where the if_ expression is lowered. When the
        # branch expression simply does `deref` on the iterator, without any shifting,
        # it corresponds to a direct element access. Such `deref` expressions can
        # be lowered outside the nested SDFG, so that just the local value (a scalar
        # or a list of values) is passed as input to the nested SDFG.
        shifted_iterator_symbols = set()
        for branch_expr in node.args[1:3]:
            for shift_node in eve.walk_values(branch_expr).filter(
                lambda x: cpm.is_applied_shift(x)
            ):
                shifted_iterator_symbols |= (
                    eve.walk_values(shift_node)
                    .if_isinstance(gtir.SymRef)
                    .map(lambda x: str(x.id))
                    .filter(lambda x: isinstance(self.symbol_map.get(x, None), IteratorExpr))
                    .to_set()
                )
        iterator_symbols = {
            sym_name
            for sym_name, sym_type in self.symbol_map.items()
            if isinstance(sym_type, IteratorExpr)
        }
        direct_deref_iterators = (
            set(symbol_ref_utils.collect_symbol_refs(node.args[1:3], iterator_symbols))
            - shifted_iterator_symbols
        )

        for nstate, arg in zip([tstate, fstate], node.args[1:3]):
            # visit each if-branch in the corresponding state of the nested SDFG
            in_edges, output_tree = self._visit_if_branch(
                nsdfg, nstate, arg, input_memlets, direct_deref_iterators
            )
            for edge in in_edges:
                edge.connect(map_entry=None)

            if isinstance(node.type, ts.TupleType):
                out_symbol_tree = gtir_sdfg_utils.make_symbol_tree("__output", node.type)
                outer_value = gtx_utils.tree_map(
                    lambda x, y, nstate=nstate: self._visit_if_branch_result(nsdfg, nstate, x, y)
                )(output_tree, out_symbol_tree)
            else:
                assert isinstance(node.type, ts.FieldType | ts.ScalarType)
                assert len(output_tree) == 1 and isinstance(output_tree[0], DataflowOutputEdge)
                output_edge = output_tree[0]
                outer_value = self._visit_if_branch_result(
                    nsdfg, nstate, output_edge, im.sym("__output", node.type)
                )
            # Isolated access node will make validation fail.
            # Isolated access nodes can be found in `make_tuple` expressions that
            # construct tuples from input arguments.
            for data_node in nstate.data_nodes():
                if nstate.degree(data_node) == 0:
                    assert not data_node.desc(nsdfg).transient
                    nsdfg.remove_node(data_node)
        else:
            result = outer_value

        outputs = {outval.dc_node.data for outval in gtx_utils.flatten_nested_tuple((result,))}

        # all free symbols are mapped to the symbols available in parent SDFG
        nsdfg_symbols_mapping = {str(sym): sym for sym in nsdfg.free_symbols}
        if isinstance(condition_value, SymbolExpr):
            nsdfg_symbols_mapping["__cond"] = condition_value.value
        nsdfg_node = self.state.add_nested_sdfg(
            nsdfg,
            self.sdfg,
            inputs=set(input_memlets.keys()),
            outputs=outputs,
            symbol_mapping=nsdfg_symbols_mapping,
        )

        for inner, input_expr in input_memlets.items():
            if isinstance(input_expr, MemletExpr):
                self._add_input_data_edge(input_expr.dc_node, input_expr.subset, nsdfg_node, inner)
            else:
                self._add_edge(
                    input_expr.dc_node,
                    None,
                    nsdfg_node,
                    inner,
                    self.sdfg.make_array_memlet(input_expr.dc_node.data),
                )

        return (
            gtx_utils.tree_map(write_output_of_nested_sdfg_to_temporary)(result)
            if isinstance(result, tuple)
            else write_output_of_nested_sdfg_to_temporary(result)
        )

    def _visit_neighbors(self, node: gtir.FunCall) -> ValueExpr:
        assert isinstance(node.type, ts.ListType)
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
        connectivity = gtx_dace_utils.connectivity_identifier(offset)
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
                subset=dace_subsets.Range.from_string(
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
                subset=dace_subsets.Range.from_string(
                    f"{origin_index.value}, 0:{offset_provider.max_neighbors}"
                ),
            )
        )

        neighbors_temp, _ = self.subgraph_builder.add_temp_array(
            self.sdfg, (offset_provider.max_neighbors,), field_desc.dtype
        )
        neighbors_node = self.state.add_access(neighbors_temp)
        offset_type = gtx_common.Dimension(offset, gtx_common.DimensionKind.LOCAL)
        neighbor_idx = gtir_sdfg_utils.get_map_variable(offset_type)

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
            dc_node=neighbors_node, gt_dtype=ts.ListType(node.type.element_type, offset_type)
        )

    def _visit_list_get(self, node: gtir.FunCall) -> ValueExpr:
        assert len(node.args) == 2
        index_arg = self.visit(node.args[0])
        list_arg = self.visit(node.args[1])
        assert isinstance(list_arg, ValueExpr)
        assert isinstance(list_arg.gt_dtype, ts.ListType)
        assert isinstance(list_arg.gt_dtype.element_type, ts.ScalarType)

        list_desc = list_arg.dc_node.desc(self.sdfg)
        assert len(list_desc.shape) == 1

        result_dtype = gtx_dace_utils.as_dace_type(list_arg.gt_dtype.element_type)
        result, _ = self.subgraph_builder.add_temp_scalar(self.sdfg, result_dtype)
        result_node = self.state.add_access(result)

        if isinstance(index_arg, SymbolExpr):
            assert index_arg.dc_dtype in dace.dtypes.INTEGER_TYPES
            self._add_edge(
                list_arg.dc_node,
                None,
                result_node,
                None,
                dace.Memlet(data=list_arg.dc_node.data, subset=index_arg.value),
            )
        elif isinstance(index_arg, ValueExpr):
            tasklet_node = self._add_tasklet(
                "list_get", inputs={"index", "list"}, outputs={"value"}, code="value = list[index]"
            )
            self._add_edge(
                index_arg.dc_node,
                None,
                tasklet_node,
                "index",
                dace.Memlet(data=index_arg.dc_node.data, subset="0"),
            )
            self._add_edge(
                list_arg.dc_node,
                None,
                tasklet_node,
                "list",
                self.sdfg.make_array_memlet(list_arg.dc_node.data),
            )
            self._add_edge(
                tasklet_node, "value", result_node, None, dace.Memlet(data=result, subset="0")
            )
        else:
            raise TypeError(f"Unexpected value {index_arg} as index argument.")

        return ValueExpr(dc_node=result_node, gt_dtype=list_arg.gt_dtype.element_type)

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
        assert isinstance(node.type, ts.ListType)
        assert isinstance(node.fun, gtir.FunCall)
        assert len(node.fun.args) == 1  # the operation to be mapped on the arguments

        assert isinstance(node.type.element_type, ts.ScalarType)
        dc_dtype = gtx_dace_utils.as_dace_type(node.type.element_type)

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
            assert isinstance(input_arg.gt_dtype, ts.ListType)
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
        map_index = gtir_sdfg_utils.get_map_variable(offset_type)

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

        result, _ = self.subgraph_builder.add_temp_array(self.sdfg, (local_size,), dc_dtype)
        result_node = self.state.add_access(result)

        if offset_provider_type.has_skip_values:
            # In case the `map_` input expressions contain skip values, we use
            # the connectivity-based offset provider as mask for map computation.
            connectivity = gtx_dace_utils.connectivity_identifier(offset_type.value)
            connectivity_desc = self.sdfg.arrays[connectivity]
            connectivity_desc.transient = False

            origin_map_index = gtir_sdfg_utils.get_map_variable(offset_provider_type.source_dim)

            connectivity_slice = self._construct_local_view(
                MemletExpr(
                    dc_node=self.state.add_access(connectivity),
                    gt_dtype=ts.ListType(
                        element_type=node.type.element_type, offset_type=offset_type
                    ),
                    subset=dace_subsets.Range.from_string(
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
            gt_dtype=ts.ListType(node.type.element_type, offset_type),
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
        origin_map_index = gtir_sdfg_utils.get_map_variable(offset_provider_type.source_dim)

        assert (
            isinstance(input_expr.gt_dtype, ts.ListType)
            and input_expr.gt_dtype.offset_type is not None
        )
        offset_type = input_expr.gt_dtype.offset_type
        connectivity = gtx_dace_utils.connectivity_identifier(offset_type.value)
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
            dace_subsets.Range.from_string(
                f"{origin_map_index}, 0:{offset_provider_type.max_neighbors}"
            ),
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

        result, _ = self.subgraph_builder.add_temp_scalar(self.sdfg, reduce_identity.dc_dtype)
        result_node = self.state.add_access(result)

        input_expr = self.visit(node.args[0])
        assert isinstance(input_expr, (MemletExpr, ValueExpr))
        assert (
            isinstance(input_expr.gt_dtype, ts.ListType)
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
            dace_subsets.Range.from_array(offset_table_node.desc(self.sdfg)),
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
        origin_dim = connectivity.source_dim
        origin_index = it.indices[origin_dim]
        assert isinstance(origin_index, SymbolExpr)

        shifted_indices = {dim: idx for dim, idx in it.indices.items() if dim != origin_dim}
        if isinstance(offset_expr, SymbolExpr):
            # use memlet to retrieve the neighbor index
            shifted_indices[neighbor_dim] = MemletExpr(
                dc_node=offset_table_node,
                gt_dtype=it.gt_dtype,
                subset=dace_subsets.Range.from_string(f"{origin_index.value}, {offset_expr.value}"),
            )
        else:
            # dynamic offset: we cannot use a memlet to retrieve the offset value, use a tasklet node
            shifted_indices[neighbor_dim] = self._make_dynamic_neighbor_offset(
                offset_expr, offset_table_node, origin_index
            )

        return IteratorExpr(it.field, it.gt_dtype, it.field_domain, shifted_indices)

    def _visit_shift(self, node: gtir.FunCall) -> IteratorExpr:
        # convert builtin-index type to dace type
        IndexDType: Final = gtx_dace_utils.as_dace_type(
            ts.ScalarType(kind=getattr(ts.ScalarKind, gtir_builtins.INTEGER_INDEX_BUILTIN.upper()))
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
            offset_table = gtx_dace_utils.connectivity_identifier(offset)
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

        if isinstance(node.type, ts.ListType):
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
            dc_dtype = gtx_dace_utils.as_dace_type(node.type.element_type)
            # In order to ease the lowring of the parent expression on local dimension,
            # we represent the scalar value as a single-element 1D array.
            use_array = True
        else:
            assert isinstance(node.type, ts.ScalarType)
            dc_dtype = gtx_dace_utils.as_dace_type(node.type)
            use_array = False

        return self._construct_tasklet_result(dc_dtype, tasklet_node, "result", use_array=use_array)

    def _visit_make_tuple(self, node: gtir.FunCall) -> tuple[IteratorExpr | DataExpr]:
        assert cpm.is_call_to(node, "make_tuple")
        return tuple(self.visit(arg) for arg in node.args)

    def _visit_tuple_get(
        self, node: gtir.FunCall
    ) -> IteratorExpr | DataExpr | tuple[IteratorExpr | DataExpr]:
        assert cpm.is_call_to(node, "tuple_get")
        assert len(node.args) == 2

        if not isinstance(node.args[0], gtir.Literal):
            raise ValueError("Tuple can only be subscripted with compile-time constants.")
        assert ti.is_integral(node.args[0].type)
        index = int(node.args[0].value)

        tuple_fields = self.visit(node.args[1])
        return tuple_fields[index]

    def visit_FunCall(
        self, node: gtir.FunCall
    ) -> IteratorExpr | DataExpr | tuple[IteratorExpr | DataExpr | tuple[Any, ...], ...]:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node, "if_"):
            return self._visit_if(node)

        elif cpm.is_call_to(node, "neighbors"):
            return self._visit_neighbors(node)

        elif cpm.is_call_to(node, "list_get"):
            return self._visit_list_get(node)

        elif cpm.is_call_to(node, "make_tuple"):
            return self._visit_make_tuple(node)

        elif cpm.is_call_to(node, "tuple_get"):
            return self._visit_tuple_get(node)

        elif cpm.is_applied_map(node):
            return self._visit_map(node)

        elif cpm.is_applied_reduce(node):
            return self._visit_reduce(node)

        elif cpm.is_applied_shift(node):
            return self._visit_shift(node)

        elif isinstance(node.fun, gtir.Lambda):
            # Lambda node should be visited with 'visit_let()' method.
            raise ValueError(f"Unexpected lambda in 'FunCall' node: {node}.")

        elif isinstance(node.fun, gtir.SymRef):
            return self._visit_generic_builtin(node)

        else:
            raise NotImplementedError(f"Invalid 'FunCall' node: {node}.")

    def visit_Lambda(
        self, node: gtir.Lambda
    ) -> DataflowOutputEdge | tuple[DataflowOutputEdge | tuple[Any, ...], ...]:
        def _visit_Lambda_impl(
            output_expr: DataflowOutputEdge | ValueExpr | MemletExpr | SymbolExpr,
        ) -> DataflowOutputEdge:
            if isinstance(output_expr, DataflowOutputEdge):
                return output_expr
            if isinstance(output_expr, ValueExpr):
                return DataflowOutputEdge(self.state, output_expr)

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
                # even simpler case, where a constant value is written to destination node
                output_dtype = output_expr.dc_dtype
                tasklet_node = self._add_tasklet(
                    "write", {}, {"__out"}, f"__out = {output_expr.value}"
                )

            output_expr = self._construct_tasklet_result(output_dtype, tasklet_node, "__out")
            return DataflowOutputEdge(self.state, output_expr)

        result = self.visit(node.expr)

        return (
            gtx_utils.tree_map(_visit_Lambda_impl)(result)
            if isinstance(result, tuple)
            else _visit_Lambda_impl(result)
        )

    def visit_Literal(self, node: gtir.Literal) -> SymbolExpr:
        dc_dtype = gtx_dace_utils.as_dace_type(node.type)
        return SymbolExpr(node.value, dc_dtype)

    def visit_SymRef(
        self, node: gtir.SymRef
    ) -> IteratorExpr | DataExpr | tuple[IteratorExpr | DataExpr | tuple[Any, ...], ...]:
        param = str(node.id)
        if param in self.symbol_map:
            return self.symbol_map[param]
        # if not in the lambda symbol map, this must be a symref to a builtin function
        assert param in gtir_python_codegen.MATH_BUILTINS_MAPPING
        return SymbolExpr(param, dace.string)

    def visit_let(
        self,
        node: gtir.Lambda,
        args: Sequence[
            IteratorExpr
            | MemletExpr
            | ValueExpr
            | tuple[IteratorExpr | MemletExpr | ValueExpr | tuple[Any, ...], ...]
        ],
    ) -> DataflowOutputEdge | tuple[DataflowOutputEdge | tuple[Any, ...], ...]:
        """
        Maps lambda arguments to internal parameters.

        This method is responsible to recognize the usage of the `Lambda` node,
        which can be either a let-statement or the stencil expression in local view.
        The usage of a `Lambda` as let-statement corresponds to computing some results
        and making them available inside the lambda scope, represented as a nested SDFG.
        All let-statements, if any, are supposed to be encountered before the stencil
        expression. In other words, the `Lambda` node representing the stencil expression
        is always the innermost node.
        Therefore, the lowering of let-statements results in recursive calls to
        `visit_let()` until the stencil expression is found. At that point, it falls
        back to the `visit()` function.
        """

        # lambda arguments are mapped to symbols defined in lambda scope.
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg

        if cpm.is_let(node.expr):
            let_node = node.expr
            let_args = [self.visit(arg) for arg in let_node.args]
            assert isinstance(let_node.fun, gtir.Lambda)
            return self.visit_let(let_node.fun, args=let_args)
        else:
            # this lambda node is not a let-statement, but a stencil expression
            return self.visit(node)


def translate_lambda_to_dataflow(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    sdfg_builder: gtir_sdfg.DataflowBuilder,
    node: gtir.Lambda,
    args: Sequence[
        IteratorExpr
        | MemletExpr
        | ValueExpr
        | tuple[IteratorExpr | MemletExpr | ValueExpr | tuple[Any, ...], ...]
    ],
) -> tuple[
    list[DataflowInputEdge],
    tuple[DataflowOutputEdge | tuple[Any, ...], ...],
]:
    """
    Entry point to visit a `Lambda` node and lower it to a dataflow graph,
    that can be instantiated inside a map scope implementing the field operator.

    It calls `LambdaToDataflow.visit_let()` to map the lambda arguments to internal
    parameters and visit the let-statements (if any), which always appear as outermost
    nodes. Finally, the visitor returns the output edge of the dataflow.

    Args:
        sdfg: The SDFG where the dataflow graph will be instantiated.
        state: The SDFG state where the dataflow graph will be instantiated.
        sdfg_builder: Helper class to build the dataflow inside the given SDFG.
        node: Lambda node to visit.
        args: Arguments passed to lambda node.

    Returns:
        A tuple of two elements:
        - List of connections for data inputs to the dataflow.
        - Tree representation of output data connections.
    """
    taskgen = LambdaToDataflow(sdfg, state, sdfg_builder)
    lambda_output = taskgen.visit_let(node, args)

    if isinstance(lambda_output, DataflowOutputEdge):
        return taskgen.input_edges, (lambda_output,)
    else:
        return taskgen.input_edges, lambda_output
