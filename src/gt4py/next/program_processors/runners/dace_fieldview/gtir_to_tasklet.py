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


import itertools
from dataclasses import dataclass
from typing import Optional, TypeAlias

import dace
import dace.subsets as sbs

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util
from gt4py.next.program_processors.runners.dace_fieldview.gtir_python_codegen import (
    MATH_BUILTINS_MAPPING,
)


@dataclass(frozen=True)
class MemletExpr:
    """Scalar or array data access thorugh a memlet."""

    source: dace.nodes.AccessNode
    subset: sbs.Indices | sbs.Range


@dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymExpr
    dtype: dace.typeclass


@dataclass(frozen=True)
class TaskletExpr:
    """Result of the computation provided by a tasklet node."""

    node: dace.nodes.Tasklet
    connector: str


IteratorIndexExpr: TypeAlias = MemletExpr | SymbolExpr | TaskletExpr


@dataclass(frozen=True)
class IteratorExpr:
    """Iterator for field access to be consumed by `deref` or `shift` builtin functions."""

    field: dace.nodes.AccessNode
    dimensions: list[str]
    indices: dict[str, IteratorIndexExpr]


InputConnection: TypeAlias = tuple[
    dace.nodes.AccessNode,
    sbs.Range,
    dace.nodes.Tasklet,
    str,
]

INDEX_CONNECTOR_FMT = "__index_{dim}"


class GTIRToTasklet(eve.NodeVisitor):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    sdfg: dace.SDFG
    state: dace.SDFGState
    input_connections: list[InputConnection]
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_map: dict[str, SymbolExpr | IteratorExpr | MemletExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        self.sdfg = sdfg
        self.state = state
        self.input_connections = []
        self.offset_provider = offset_provider
        self.symbol_map = {}

    def _add_input_connection(
        self,
        src: dace.nodes.AccessNode,
        subset: sbs.Range,
        dst: dace.nodes.Tasklet,
        dst_connector: str,
    ) -> None:
        self.input_connections.append((src, subset, dst, dst_connector))

    def _visit_deref(self, node: itir.FunCall) -> MemletExpr | TaskletExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, IteratorExpr):
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # use direct field access through memlet subset
                data_index = sbs.Indices([it.indices[dim].value for dim in it.dimensions])  # type: ignore[union-attr]
                return MemletExpr(it.field, data_index)

            else:
                assert all(dim in it.indices.keys() for dim in it.dimensions)
                field_indices = [(dim, it.indices[dim]) for dim in it.dimensions]
                index_connectors = [
                    INDEX_CONNECTOR_FMT.format(dim=dim)
                    for dim, index in field_indices
                    if not isinstance(index, SymbolExpr)
                ]
                index_internals = ",".join(
                    str(index.value)
                    if isinstance(index, SymbolExpr)
                    else INDEX_CONNECTOR_FMT.format(dim=dim)
                    for dim, index in field_indices
                )
                deref_node = self.state.add_tasklet(
                    "deref_field_indirection",
                    {"field"} | set(index_connectors),
                    {"val"},
                    code=f"val = field[{index_internals}]",
                )
                # add new termination point for this field parameter
                field_desc = it.field.desc(self.sdfg)
                field_fullset = sbs.Range.from_array(field_desc)
                self._add_input_connection(it.field, field_fullset, deref_node, "field")

                for dim, index_expr in field_indices:
                    deref_connector = INDEX_CONNECTOR_FMT.format(dim=dim)
                    if isinstance(index_expr, MemletExpr):
                        self._add_input_connection(
                            index_expr.source,
                            index_expr.subset,
                            deref_node,
                            deref_connector,
                        )

                    elif isinstance(index_expr, TaskletExpr):
                        self.state.add_edge(
                            index_expr.node,
                            index_expr.connector,
                            deref_node,
                            deref_connector,
                            dace.Memlet(),
                        )
                    else:
                        assert isinstance(index_expr, SymbolExpr)

                return TaskletExpr(deref_node, "val")

        else:
            assert isinstance(it, MemletExpr)
            return it

    def _split_shift_args(
        self, args: list[itir.Expr]
    ) -> tuple[list[itir.Expr], Optional[list[itir.Expr]]]:
        pairs = [args[i : i + 2] for i in range(0, len(args), 2)]
        assert len(pairs) >= 1
        assert all(len(pair) == 2 for pair in pairs)
        return pairs[-1], list(itertools.chain(*pairs[0:-1])) if len(pairs) > 1 else None

    def _make_shift_for_rest(self, rest: list[itir.Expr], iterator: itir.Expr) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest),
            args=[iterator],
        )

    def _make_cartesian_shift(
        self, it: IteratorExpr, offset_dim: Dimension, offset_expr: IteratorIndexExpr
    ) -> IteratorExpr:
        """Implements cartesian offset along one dimension."""
        assert offset_dim.value in it.dimensions
        new_index: SymbolExpr | TaskletExpr
        assert offset_dim.value in it.indices
        index_expr = it.indices[offset_dim.value]
        if isinstance(index_expr, SymbolExpr) and isinstance(offset_expr, SymbolExpr):
            new_index = SymbolExpr(index_expr.value + offset_expr.value, index_expr.dtype)
        else:
            # the offset needs to be calculate by means of a tasklet
            new_index_connector = "shifted_index"
            if isinstance(index_expr, SymbolExpr):
                dynamic_offset_tasklet = self.state.add_tasklet(
                    "dynamic_offset",
                    {"offset"},
                    {new_index_connector},
                    f"{new_index_connector} = {index_expr.value} + offset",
                )
            elif isinstance(offset_expr, SymbolExpr):
                dynamic_offset_tasklet = self.state.add_tasklet(
                    "dynamic_offset",
                    {"index"},
                    {new_index_connector},
                    f"{new_index_connector} = index + {offset_expr}",
                )
            else:
                dynamic_offset_tasklet = self.state.add_tasklet(
                    "dynamic_offset",
                    {"index", "offset"},
                    {new_index_connector},
                    f"{new_index_connector} = index + offset",
                )
            for input_expr, input_connector in [(index_expr, "index"), (offset_expr, "offset")]:
                if isinstance(input_expr, MemletExpr):
                    self._add_input_connection(
                        input_expr.source,
                        input_expr.subset,
                        dynamic_offset_tasklet,
                        input_connector,
                    )
                elif isinstance(input_expr, TaskletExpr):
                    self.state.add_edge(
                        input_expr.node,
                        input_expr.connector,
                        dynamic_offset_tasklet,
                        input_connector,
                        dace.Memlet(),
                    )

            new_index = TaskletExpr(dynamic_offset_tasklet, new_index_connector)

        return IteratorExpr(
            it.field,
            it.dimensions,
            {
                dim: (new_index if dim == offset_dim.value else index)
                for dim, index in it.indices.items()
            },
        )

    def _make_unstructured_shift(
        self,
        it: IteratorExpr,
        connectivity: Connectivity,
        offset_table_node: dace.nodes.AccessNode,
        offset_expr: IteratorIndexExpr,
    ) -> IteratorExpr:
        # shift in unstructured domain by means of a neighbor table
        neighbor_dim = connectivity.neighbor_axis.value
        assert neighbor_dim in it.dimensions

        origin_dim = connectivity.origin_axis.value
        if origin_dim in it.indices:
            origin_index = it.indices[origin_dim]
            assert isinstance(origin_index, SymbolExpr)
            neighbor_expr = it.indices.get(neighbor_dim, None)
            if neighbor_expr is not None:
                assert isinstance(neighbor_expr, TaskletExpr)
                if isinstance(offset_expr, SymbolExpr):
                    # use memlet to retrieve the neighbor index and pass it to the index connector of tasklet for neighbor access
                    self._add_input_connection(
                        offset_table_node,
                        sbs.Indices([origin_index.value, offset_expr.value]),
                        neighbor_expr.node,
                        INDEX_CONNECTOR_FMT.format(dim=neighbor_dim),
                    )
                else:
                    # dynamic offset: we cannot use a memlet to retrieve the offset value, use a tasklet node
                    dynamic_offset_tasklet = self._make_dynamic_neighbor_offset(
                        offset_expr, offset_table_node, origin_index
                    )

                    # write result to the index connector of tasklet for neighbor access
                    self.state.add_edge(
                        dynamic_offset_tasklet.node,
                        dynamic_offset_tasklet.connector,
                        neighbor_expr.node,
                        INDEX_CONNECTOR_FMT.format(dim=neighbor_dim),
                    )

                shifted_indices = {
                    dim: index for dim, index in it.indices.items() if dim != neighbor_dim
                } | {origin_dim: it.indices[neighbor_dim]}

            elif isinstance(offset_expr, SymbolExpr):
                # use memlet to retrieve the neighbor index
                shifted_indices = it.indices | {
                    neighbor_dim: MemletExpr(
                        offset_table_node,
                        sbs.Indices([origin_index.value, offset_expr.value]),
                    )
                }
            else:
                # dynamic offset: we cannot use a memlet to retrieve the offset value, use a tasklet node
                dynamic_offset_tasklet = self._make_dynamic_neighbor_offset(
                    offset_expr, offset_table_node, origin_index
                )

                shifted_indices = it.indices | {neighbor_dim: dynamic_offset_tasklet}

        else:
            origin_index_connector = INDEX_CONNECTOR_FMT.format(dim=origin_dim)
            neighbor_index_connector = INDEX_CONNECTOR_FMT.format(dim=neighbor_dim)
            if isinstance(offset_expr, SymbolExpr):
                tasklet_node = self.state.add_tasklet(
                    "shift",
                    {"table", origin_index_connector},
                    {neighbor_index_connector},
                    f"{neighbor_index_connector} = table[{origin_index_connector}, {offset_expr.value}]",
                )
            else:
                tasklet_node = self.state.add_tasklet(
                    "shift",
                    {"table", origin_index_connector, "offset"},
                    {neighbor_index_connector},
                    f"{neighbor_index_connector} = table[{origin_index_connector}, offset]",
                )
                if isinstance(offset_expr, MemletExpr):
                    self._add_input_connection(
                        offset_expr.source,
                        offset_expr.subset,
                        tasklet_node,
                        "offset",
                    )
                else:
                    self.state.add_edge(
                        offset_expr.node,
                        offset_expr.connector,
                        tasklet_node,
                        "offset",
                        dace.Memlet(),
                    )
            neighbor_expr = TaskletExpr(
                tasklet_node,
                neighbor_index_connector,
            )
            table_desc = offset_table_node.desc(self.sdfg)
            self._add_input_connection(
                offset_table_node,
                sbs.Range.from_array(table_desc),
                tasklet_node,
                "table",
            )
            shifted_indices = it.indices | {origin_dim: neighbor_expr}

        return IteratorExpr(
            it.field,
            [origin_dim if neighbor_expr and dim == neighbor_dim else dim for dim in it.dimensions],
            shifted_indices,
        )

    def _make_dynamic_neighbor_offset(
        self,
        offset_expr: MemletExpr | TaskletExpr,
        offset_table_node: dace.nodes.AccessNode,
        origin_index: SymbolExpr,
    ) -> TaskletExpr:
        new_index_connector = "neighbor_index"
        tasklet_node = self.state.add_tasklet(
            "dynamic_neighbor_offset",
            {"table", "offset"},
            {new_index_connector},
            f"{new_index_connector} = table[{origin_index.value}, offset]",
        )
        self._add_input_connection(
            offset_table_node,
            sbs.Range.from_array(offset_table_node.desc(self.sdfg)),
            tasklet_node,
            "table",
        )
        if isinstance(offset_expr, MemletExpr):
            self._add_input_connection(
                offset_expr.source,
                offset_expr.subset,
                tasklet_node,
                "offset",
            )
        else:
            self.state.add_edge(
                offset_expr.node,
                offset_expr.connector,
                tasklet_node,
                "offset",
                dace.Memlet(),
            )

        return TaskletExpr(tasklet_node, new_index_connector)

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr:
        shift_node = node.fun
        assert isinstance(shift_node, itir.FunCall)

        # the iterator to be shifted is the argument to the function node
        head, tail = self._split_shift_args(shift_node.args)
        if tail:
            it = self.visit(self._make_shift_for_rest(tail, node.args[0]))
        else:
            it = self.visit(node.args[0])
        assert isinstance(it, IteratorExpr)

        # first argument of the shift node is the offset provider
        assert isinstance(head[0], itir.OffsetLiteral)
        offset = head[0].value
        assert isinstance(offset, str)
        offset_provider = self.offset_provider[offset]
        # second argument should be the offset value, which could be a symbolic expression or a dynamic offset
        offset_expr: IteratorIndexExpr
        if isinstance(head[1], itir.OffsetLiteral):
            offset_expr = SymbolExpr(head[1].value, dace.int32)
        else:
            dynamic_offset_expr = self.visit(head[1])
            assert isinstance(dynamic_offset_expr, MemletExpr | TaskletExpr)
            offset_expr = dynamic_offset_expr

        if isinstance(offset_provider, Dimension):
            return self._make_cartesian_shift(it, offset_provider, offset_expr)
        else:
            # initially, the storage for the connectivty tables is created as transient
            # when the tables are used, the storage is changed to non-transient,
            # so the corresponding arrays are supposed to be allocated by the SDFG caller
            offset_table = dace_fieldview_util.connectivity_identifier(offset)
            self.sdfg.arrays[offset_table].transient = False
            offset_table_node = self.state.add_access(offset_table)

            return self._make_unstructured_shift(
                it, offset_provider, offset_table_node, offset_expr
            )

    def visit_FunCall(self, node: itir.FunCall) -> IteratorExpr | TaskletExpr | MemletExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node.fun, "shift"):
            return self._visit_shift(node)

        else:
            assert isinstance(node.fun, itir.SymRef)

        node_internals = []
        node_connections: dict[str, MemletExpr | TaskletExpr] = {}
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, MemletExpr | TaskletExpr):
                # the argument value is the result of a tasklet node or direct field access
                connector = f"__inp_{i}"
                node_connections[connector] = arg_expr
                node_internals.append(connector)
            else:
                assert isinstance(arg_expr, SymbolExpr)
                # use the argument value without adding any connector
                node_internals.append(arg_expr.value)

        # create a tasklet node implementing the builtin function
        builtin_name = str(node.fun.id)
        if builtin_name in MATH_BUILTINS_MAPPING:
            fmt = MATH_BUILTINS_MAPPING[builtin_name]
            code = fmt.format(*node_internals)
        else:
            raise NotImplementedError(f"'{builtin_name}' not implemented.")

        out_connector = "result"
        tasklet_node = self.state.add_tasklet(
            builtin_name,
            node_connections.keys(),
            {out_connector},
            "{} = {}".format(out_connector, code),
        )

        for connector, arg_expr in node_connections.items():
            if isinstance(arg_expr, TaskletExpr):
                self.state.add_edge(
                    arg_expr.node, arg_expr.connector, tasklet_node, connector, dace.Memlet()
                )
            else:
                self._add_input_connection(
                    arg_expr.source, arg_expr.subset, tasklet_node, connector
                )

        return TaskletExpr(tasklet_node, "result")

    def visit_Lambda(
        self, node: itir.Lambda, args: list[SymbolExpr | IteratorExpr | MemletExpr]
    ) -> tuple[
        list[InputConnection],
        TaskletExpr,
    ]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr = self.visit(node.expr)
        if isinstance(output_expr, TaskletExpr):
            return self.input_connections, output_expr

        # special case where the field operator is simply copying data from source to destination node
        assert isinstance(output_expr, MemletExpr)
        tasklet_node = self.state.add_tasklet("copy", {"__inp"}, {"__out"}, "__out = __inp")
        self._add_input_connection(output_expr.source, output_expr.subset, tasklet_node, "__inp")
        return self.input_connections, TaskletExpr(tasklet_node, "__out")

    def visit_Literal(self, node: itir.Literal) -> SymbolExpr:
        dtype = dace_fieldview_util.as_dace_type(node.type)
        return SymbolExpr(node.value, dtype)

    def visit_SymRef(self, node: itir.SymRef) -> SymbolExpr | IteratorExpr | MemletExpr:
        param = str(node.id)
        assert param in self.symbol_map
        return self.symbol_map[param]
