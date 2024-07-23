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

import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, TypeAlias, Union

import dace
import dace.subsets as sbs

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import (
    gtir_python_codegen,
    gtir_to_sdfg,
    utility as dace_fieldview_util,
)
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class MemletExpr:
    """Scalar or array data access thorugh a memlet."""

    node: dace.nodes.AccessNode
    subset: sbs.Indices | sbs.Range


@dataclasses.dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymExpr
    dtype: dace.typeclass


@dataclasses.dataclass(frozen=True)
class ValueExpr:
    """Result of the computation implemented by a tasklet node."""

    node: dace.nodes.AccessNode
    field_type: ts.FieldType | ts.ScalarType


# Define alias for the elements needed to setup input connections to a map scope
InputConnection: TypeAlias = tuple[
    dace.nodes.AccessNode,
    sbs.Range,
    dace.nodes.Node,
    Optional[str],
]

IteratorIndexExpr: TypeAlias = MemletExpr | SymbolExpr | ValueExpr


@dataclasses.dataclass(frozen=True)
class IteratorExpr:
    """Iterator for field access to be consumed by `deref` or `shift` builtin functions."""

    field: dace.nodes.AccessNode
    dimensions: list[gtx_common.Dimension]
    indices: dict[gtx_common.Dimension, IteratorIndexExpr]


class LambdaToTasklet(eve.NodeVisitor):
    """Translates an `ir.Lambda` expression to a dataflow graph.

    Lambda functions should only be encountered as argument to the `as_field_op`
    builtin function, therefore the dataflow graph generated here typically
    represents the stencil function of a field operator.
    """

    sdfg: dace.SDFG
    state: dace.SDFGState
    subgraph_builder: gtir_to_sdfg.DataflowBuilder
    input_connections: list[InputConnection]
    symbol_map: dict[str, IteratorExpr | MemletExpr | SymbolExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        subgraph_builder: gtir_to_sdfg.DataflowBuilder,
    ):
        self.sdfg = sdfg
        self.state = state
        self.subgraph_builder = subgraph_builder
        self.input_connections = []
        self.symbol_map = {}

    def _add_entry_memlet_path(
        self,
        src: dace.nodes.AccessNode,
        src_subset: sbs.Range,
        dst_node: dace.nodes.Node,
        dst_conn: Optional[str] = None,
    ) -> None:
        self.input_connections.append((src, src_subset, dst_node, dst_conn))

    def _add_map(
        self,
        name: str,
        ndrange: Union[
            Dict[str, Union[str, dace.subsets.Subset]],
            List[Tuple[str, Union[str, dace.subsets.Subset]]],
        ],
        **kwargs: Any,
    ) -> Tuple[dace.nodes.MapEntry, dace.nodes.MapExit]:
        """Helper method to add a map with unique ame in current state."""
        return self.subgraph_builder.add_map(name, self.state, ndrange, **kwargs)

    def _add_tasklet(
        self,
        name: str,
        inputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        outputs: Union[Set[str], Dict[str, dace.dtypes.typeclass]],
        code: str,
        **kwargs: Any,
    ) -> dace.nodes.Tasklet:
        """Helper method to add a tasklet with unique ame in current state."""
        return self.subgraph_builder.add_tasklet(name, self.state, inputs, outputs, code, **kwargs)

    def _get_tasklet_result(
        self,
        dtype: dace.typeclass,
        src_node: dace.nodes.Tasklet,
        src_connector: str,
    ) -> ValueExpr:
        temp_name = self.sdfg.temp_data_name()
        self.sdfg.add_scalar(temp_name, dtype, transient=True)
        data_type = dace_fieldview_util.as_scalar_type(str(dtype.as_numpy_dtype()))
        temp_node = self.state.add_access(temp_name)
        self.state.add_edge(
            src_node,
            src_connector,
            temp_node,
            None,
            dace.Memlet(data=temp_name, subset="0"),
        )
        return ValueExpr(temp_node, data_type)

    def _visit_deref(self, node: gtir.FunCall) -> MemletExpr | ValueExpr:
        # format used for field index tasklet connector
        IndexConnectorFmt = "__index_{dim}"

        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, IteratorExpr):
            field_desc = it.field.desc(self.sdfg)
            assert len(field_desc.shape) == len(it.dimensions)
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # when all indices are symblic expressions, we can perform direct field access through a memlet
                field_subset = sbs.Indices([it.indices[dim].value for dim in it.dimensions])  # type: ignore[union-attr]
                return MemletExpr(it.field, field_subset)

            else:
                # we use a tasklet to perform dereferencing of a generic iterator
                assert all(dim in it.indices for dim in it.dimensions)
                field_indices = [(dim, it.indices[dim]) for dim in it.dimensions]
                index_connectors = [
                    IndexConnectorFmt.format(dim=dim.value)
                    for dim, index in field_indices
                    if not isinstance(index, SymbolExpr)
                ]
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
                self._add_entry_memlet_path(
                    it.field,
                    sbs.Range.from_array(field_desc),
                    deref_node,
                    "field",
                )

                for dim, index_expr in field_indices:
                    # add termination points for the dynamic iterator indices
                    deref_connector = IndexConnectorFmt.format(dim=dim.value)
                    if isinstance(index_expr, MemletExpr):
                        self._add_entry_memlet_path(
                            index_expr.node,
                            index_expr.subset,
                            deref_node,
                            deref_connector,
                        )

                    elif isinstance(index_expr, ValueExpr):
                        self.state.add_edge(
                            index_expr.node,
                            None,
                            deref_node,
                            deref_connector,
                            dace.Memlet(data=index_expr.node.data, subset="0"),
                        )
                    else:
                        assert isinstance(index_expr, SymbolExpr)

                dtype = it.field.desc(self.sdfg).dtype
                return self._get_tasklet_result(dtype, deref_node, "val")

        else:
            assert isinstance(it, MemletExpr)
            return it

    def _split_shift_args(
        self, args: list[gtir.Expr]
    ) -> tuple[list[gtir.Expr], Optional[list[gtir.Expr]]]:
        """
        Splits the arguments to `shift` builtin function as pairs, each pair containing
        the offset provider and the offset value in one dimension.
        """
        pairs = [args[i : i + 2] for i in range(0, len(args), 2)]
        assert len(pairs) >= 1
        assert all(len(pair) == 2 for pair in pairs)
        return pairs[-1], list(itertools.chain(*pairs[0:-1])) if len(pairs) > 1 else None

    def _make_shift_for_rest(self, rest: list[gtir.Expr], iterator: gtir.Expr) -> gtir.FunCall:
        """Transforms a multi-dimensional shift into recursive shift calls, each in a single dimension."""
        return gtir.FunCall(
            fun=gtir.FunCall(fun=gtir.SymRef(id="shift"), args=rest),
            args=[iterator],
        )

    def _make_cartesian_shift(
        self, it: IteratorExpr, offset_dim: gtx_common.Dimension, offset_expr: IteratorIndexExpr
    ) -> IteratorExpr:
        """Implements cartesian shift along one dimension."""
        assert offset_dim in it.dimensions
        new_index: SymbolExpr | ValueExpr
        assert offset_dim in it.indices
        index_expr = it.indices[offset_dim]
        if isinstance(index_expr, SymbolExpr) and isinstance(offset_expr, SymbolExpr):
            # purely symbolic expression which can be interpreted at compile time
            new_index = SymbolExpr(
                dace.symbolic.SymExpr(index_expr.value) + offset_expr.value, index_expr.dtype
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
                    if input_connector == "index":
                        dtype = input_expr.node.desc(self.sdfg).dtype
                    self._add_entry_memlet_path(
                        input_expr.node,
                        input_expr.subset,
                        dynamic_offset_tasklet,
                        input_connector,
                    )
                elif isinstance(input_expr, ValueExpr):
                    if input_connector == "index":
                        dtype = input_expr.node.desc(self.sdfg).dtype
                    self.state.add_edge(
                        input_expr.node,
                        None,
                        dynamic_offset_tasklet,
                        input_connector,
                        dace.Memlet(data=input_expr.node.data, subset="0"),
                    )
                else:
                    assert isinstance(input_expr, SymbolExpr)
                    if input_connector == "index":
                        dtype = input_expr.dtype

            new_index = self._get_tasklet_result(dtype, dynamic_offset_tasklet, new_index_connector)

        # a new iterator with a shifted index along one dimension
        return IteratorExpr(
            it.field,
            it.dimensions,
            {dim: (new_index if dim == offset_dim else index) for dim, index in it.indices.items()},
        )

    def _make_dynamic_neighbor_offset(
        self,
        offset_expr: MemletExpr | ValueExpr,
        offset_table_node: dace.nodes.AccessNode,
        origin_index: SymbolExpr,
    ) -> ValueExpr:
        """
        Implements access to neighbor connectivity table by means of a tasklet node.

        It requires a dynamic offset value, either obtained from a field/scalar argument (`MemletExpr`)
        or computed by another tasklet (`ValueExpr`).
        """
        new_index_connector = "neighbor_index"
        tasklet_node = self._add_tasklet(
            "dynamic_neighbor_offset",
            {"table", "offset"},
            {new_index_connector},
            f"{new_index_connector} = table[{origin_index.value}, offset]",
        )
        self._add_entry_memlet_path(
            offset_table_node,
            sbs.Range.from_array(offset_table_node.desc(self.sdfg)),
            tasklet_node,
            "table",
        )
        if isinstance(offset_expr, MemletExpr):
            self._add_entry_memlet_path(
                offset_expr.node,
                offset_expr.subset,
                tasklet_node,
                "offset",
            )
        else:
            self.state.add_edge(
                offset_expr.node,
                None,
                tasklet_node,
                "offset",
                dace.Memlet(data=offset_expr.node.data, subset="0"),
            )

        dtype = offset_table_node.desc(self.sdfg).dtype
        return self._get_tasklet_result(dtype, tasklet_node, new_index_connector)

    def _make_unstructured_shift(
        self,
        it: IteratorExpr,
        connectivity: gtx_common.Connectivity,
        offset_table_node: dace.nodes.AccessNode,
        offset_expr: IteratorIndexExpr,
    ) -> IteratorExpr:
        """Implements shift in unstructured domain by means of a neighbor table."""
        assert connectivity.neighbor_axis in it.dimensions
        neighbor_dim = connectivity.neighbor_axis
        assert neighbor_dim not in it.indices

        origin_dim = connectivity.origin_axis
        assert origin_dim in it.indices
        origin_index = it.indices[origin_dim]
        assert isinstance(origin_index, SymbolExpr)

        if isinstance(offset_expr, SymbolExpr):
            # use memlet to retrieve the neighbor index
            shifted_indices = it.indices | {
                neighbor_dim: MemletExpr(
                    offset_table_node,
                    sbs.Indices([origin_index.value, offset_expr.value]),
                )
            }
        else:
            # dynamic offset: we cannot use a memlet to retrieve the offset value, use a tasklet node
            dynamic_offset_value = self._make_dynamic_neighbor_offset(
                offset_expr, offset_table_node, origin_index
            )

            shifted_indices = it.indices | {neighbor_dim: dynamic_offset_value}

        shifted_indices.pop(origin_dim)
        return IteratorExpr(it.field, it.dimensions, shifted_indices)

    def _visit_shift(self, node: gtir.FunCall) -> IteratorExpr:
        shift_node = node.fun
        assert isinstance(shift_node, gtir.FunCall)

        # here we check the arguments to the `shift` builtin function: the offset provider and the offset value
        head, tail = self._split_shift_args(shift_node.args)
        if tail:
            # we visit a multi-dimensional shift as recursive shift function calls, each returning a new iterator
            it = self.visit(self._make_shift_for_rest(tail, node.args[0]))
        else:
            # the iterator to be shifted is the argument to the function node
            it = self.visit(node.args[0])
        assert isinstance(it, IteratorExpr)

        # first argument of the shift node is the offset provider
        assert isinstance(head[0], gtir.OffsetLiteral)
        offset = head[0].value
        assert isinstance(offset, str)
        offset_provider = self.subgraph_builder.get_offset_provider(offset)
        # second argument should be the offset value, which could be a symbolic expression or a dynamic offset
        offset_expr: IteratorIndexExpr
        if isinstance(head[1], gtir.OffsetLiteral):
            offset_expr = SymbolExpr(head[1].value, dace.int32)
        else:
            dynamic_offset_expr = self.visit(head[1])
            assert isinstance(dynamic_offset_expr, MemletExpr | ValueExpr)
            offset_expr = dynamic_offset_expr

        if isinstance(offset_provider, gtx_common.Dimension):
            return self._make_cartesian_shift(it, offset_provider, offset_expr)
        else:
            # initially, the storage for the connectivity tables is created as transient;
            # when the tables are used, the storage is changed to non-transient,
            # so the corresponding arrays are supposed to be allocated by the SDFG caller
            offset_table = dace_fieldview_util.connectivity_identifier(offset)
            self.sdfg.arrays[offset_table].transient = False
            offset_table_node = self.state.add_access(offset_table)

            return self._make_unstructured_shift(
                it, offset_provider, offset_table_node, offset_expr
            )

    def visit_FunCall(self, node: gtir.FunCall) -> IteratorExpr | MemletExpr | ValueExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_applied_shift(node):
            return self._visit_shift(node)

        else:
            assert isinstance(node.fun, gtir.SymRef)

        node_internals = []
        node_connections: dict[str, MemletExpr | ValueExpr] = {}
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, MemletExpr | ValueExpr):
                # the argument value is the result of a tasklet node or direct field access
                connector = f"__inp_{i}"
                node_connections[connector] = arg_expr
                node_internals.append(connector)
            else:
                assert isinstance(arg_expr, SymbolExpr)
                # use the argument value without adding any connector
                node_internals.append(arg_expr.value)

        # use tasklet connectors as expression arguments
        builtin_name = str(node.fun.id)
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
                self.state.add_edge(
                    arg_expr.node,
                    None,
                    tasklet_node,
                    connector,
                    dace.Memlet(data=arg_expr.node.data, subset="0"),
                )
            else:
                self._add_entry_memlet_path(
                    arg_expr.node,
                    arg_expr.subset,
                    tasklet_node,
                    connector,
                )

        assert node.type
        dtype = dace_fieldview_util.as_dace_type(node.type)

        return self._get_tasklet_result(dtype, tasklet_node, "result")

    def visit_Lambda(
        self, node: gtir.Lambda, args: list[IteratorExpr | MemletExpr | SymbolExpr]
    ) -> tuple[list[InputConnection], ValueExpr]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr: MemletExpr | SymbolExpr | ValueExpr = self.visit(node.expr)
        if isinstance(output_expr, ValueExpr):
            return self.input_connections, output_expr

        if isinstance(output_expr, MemletExpr):
            # special case where the field operator is simply copying data from source to destination node
            output_dtype = output_expr.node.desc(self.sdfg).dtype
            tasklet_node = self._add_tasklet("copy", {"__inp"}, {"__out"}, "__out = __inp")
            self._add_entry_memlet_path(
                output_expr.node,
                output_expr.subset,
                tasklet_node,
                "__inp",
            )
        else:
            # even simpler case, where a constant value is written to destination node
            output_dtype = output_expr.dtype
            tasklet_node = self._add_tasklet("write", {}, {"__out"}, f"__out = {output_expr.value}")
        return self.input_connections, self._get_tasklet_result(output_dtype, tasklet_node, "__out")

    def visit_Literal(self, node: gtir.Literal) -> SymbolExpr:
        dtype = dace_fieldview_util.as_dace_type(node.type)
        return SymbolExpr(node.value, dtype)

    def visit_SymRef(self, node: gtir.SymRef) -> IteratorExpr | MemletExpr | SymbolExpr:
        param = str(node.id)
        assert param in self.symbol_map
        return self.symbol_map[param]
