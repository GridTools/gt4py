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
import numpy as np

from gt4py import eve
from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class MemletExpr:
    """Scalar or array data access thorugh a memlet."""

    node: dace.nodes.AccessNode
    subset: sbs.Indices | sbs.Range


@dataclass(frozen=True)
class SymbolExpr:
    """Any symbolic expression that is constant in the context of current SDFG."""

    value: dace.symbolic.SymExpr
    dtype: dace.typeclass


@dataclass(frozen=True)
class ValueExpr:
    """Result of the computation implemented by a tasklet node."""

    node: dace.nodes.AccessNode


IteratorIndexExpr: TypeAlias = MemletExpr | SymbolExpr | ValueExpr


@dataclass(frozen=True)
class IteratorExpr:
    """Iterator for field access to be consumed by `deref` or `shift` builtin functions."""

    field: dace.nodes.AccessNode
    dimensions: list[str]
    indices: dict[str, IteratorIndexExpr]


# Define alias for the elements needed to setup input connections to a map scope
InputConnection: TypeAlias = tuple[
    dace.nodes.AccessNode,
    sbs.Range,
    dace.nodes.Node,
    Optional[str],
]

INDEX_CONNECTOR_FMT = "__index_{dim}"


MATH_BUILTINS_MAPPING = {
    "abs": "abs({})",
    "sin": "math.sin({})",
    "cos": "math.cos({})",
    "tan": "math.tan({})",
    "arcsin": "asin({})",
    "arccos": "acos({})",
    "arctan": "atan({})",
    "sinh": "math.sinh({})",
    "cosh": "math.cosh({})",
    "tanh": "math.tanh({})",
    "arcsinh": "asinh({})",
    "arccosh": "acosh({})",
    "arctanh": "atanh({})",
    "sqrt": "math.sqrt({})",
    "exp": "math.exp({})",
    "log": "math.log({})",
    "gamma": "tgamma({})",
    "cbrt": "cbrt({})",
    "isfinite": "isfinite({})",
    "isinf": "isinf({})",
    "isnan": "isnan({})",
    "floor": "math.ifloor({})",
    "ceil": "ceil({})",
    "trunc": "trunc({})",
    "minimum": "min({}, {})",
    "maximum": "max({}, {})",
    "fmod": "fmod({}, {})",
    "power": "math.pow({}, {})",
    "float": "dace.float64({})",
    "float32": "dace.float32({})",
    "float64": "dace.float64({})",
    "int": "dace.int32({})" if np.dtype(int).itemsize == 4 else "dace.int64({})",
    "int32": "dace.int32({})",
    "int64": "dace.int64({})",
    "bool": "dace.bool_({})",
    "plus": "({} + {})",
    "minus": "({} - {})",
    "multiplies": "({} * {})",
    "divides": "({} / {})",
    "floordiv": "({} // {})",
    "eq": "({} == {})",
    "not_eq": "({} != {})",
    "less": "({} < {})",
    "less_equal": "({} <= {})",
    "greater": "({} > {})",
    "greater_equal": "({} >= {})",
    "and_": "({} & {})",
    "or_": "({} | {})",
    "xor_": "({} ^ {})",
    "mod": "({} % {})",
    "not_": "(not {})",  # ~ is not bitwise in numpy
}


def build_neighbors_sdfg(
    field_dtype: dace.typeclass,
    field_shape: tuple[int],
    neighbors_shape: tuple[int],
    index_dtype: dace.typeclass,
) -> tuple[dace.SDFG, str, str, str]:
    assert len(field_shape) == len(neighbors_shape)

    sdfg = dace.SDFG("neighbors")
    state = sdfg.add_state()
    me, mx = state.add_map(
        "neighbors",
        {f"__idx_{i}": sbs.Range([(0, size - 1, 1)]) for i, size in enumerate(neighbors_shape)},
    )
    neighbor_index = ",".join(f"__idx_{i}" for i in range(len(neighbors_shape)))

    field_name, field_array = sdfg.add_array("field", field_shape, field_dtype)
    index_name, _ = sdfg.add_array("indexes", neighbors_shape, index_dtype)
    var_name, _ = sdfg.add_array("values", neighbors_shape, field_dtype)
    tasklet_node = state.add_tasklet(
        "gather_neighbors",
        {"__field", "__index"},
        {"__val"},
        "__val = __field[__index]",
    )
    state.add_memlet_path(
        state.add_access(field_name),
        me,
        tasklet_node,
        dst_conn="__field",
        memlet=dace.Memlet.from_array(field_name, field_array),
    )
    state.add_memlet_path(
        state.add_access(index_name),
        me,
        tasklet_node,
        dst_conn="__index",
        memlet=dace.Memlet(data=index_name, subset=neighbor_index),
    )
    state.add_memlet_path(
        tasklet_node,
        mx,
        state.add_access(var_name),
        src_conn="__val",
        memlet=dace.Memlet(data=var_name, subset=neighbor_index),
    )

    return sdfg, field_name, index_name, var_name


class LambdaToTasklet(eve.NodeVisitor):
    """Translates an `ir.Lambda` expression to a dataflow graph.

    Lambda functions should only be encountered as argument to the `as_field_op`
    builtin function, therefore the dataflow graph generated here typically
    represents the stencil function of a field operator.
    """

    sdfg: dace.SDFG
    state: dace.SDFGState
    input_connections: list[InputConnection]
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_map: dict[str, IteratorExpr | MemletExpr | SymbolExpr]

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
        dst: dace.nodes.Node,
        dst_connector: Optional[str] = None,
    ) -> None:
        self.input_connections.append((src, subset, dst, dst_connector))

    def _get_tasklet_result(
        self,
        dtype: dace.typeclass,
        src_node: dace.nodes.Node,
        src_connector: Optional[str] = None,
        subset: Optional[sbs.Range] = None,
    ) -> ValueExpr:
        if subset:
            var_name, _ = self.sdfg.add_array(
                "var", subset.size(), dtype, transient=True, find_new_name=True
            )
        else:
            var_name, _ = self.sdfg.add_scalar("var", dtype, transient=True, find_new_name=True)
            subset = "0"
        var_node = self.state.add_access(var_name)
        self.state.add_edge(
            src_node,
            src_connector,
            var_node,
            None,
            dace.Memlet(data=var_node.data, subset=subset),
        )
        return ValueExpr(var_node)

    def _visit_deref(self, node: itir.FunCall) -> MemletExpr | ValueExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, IteratorExpr):
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # when all indices are symblic expressions, we can perform direct field access through a memlet
                data_index = sbs.Indices([it.indices[dim].value for dim in it.dimensions])  # type: ignore[union-attr]
                return MemletExpr(it.field, data_index)

            else:
                # we use a tasklet to perform dereferencing of a generic iterator
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

    def _visit_neighbors(self, node: itir.FunCall) -> ValueExpr:
        assert len(node.args) == 2

        assert isinstance(node.args[0], itir.OffsetLiteral)
        offset = node.args[0].value
        assert isinstance(offset, str)
        offset_provider = self.offset_provider[offset]
        assert isinstance(offset_provider, Connectivity)

        it = self.visit(node.args[1])
        assert isinstance(it, IteratorExpr)
        assert offset_provider.neighbor_axis.value in it.dimensions
        assert offset_provider.origin_axis.value in it.indices
        origin_index = it.indices[offset_provider.origin_axis.value]
        assert isinstance(origin_index, SymbolExpr)
        assert offset_provider.origin_axis.value not in it.dimensions
        assert all(isinstance(index, SymbolExpr) for index in it.indices.values())

        field_desc = it.field.desc(self.sdfg)
        offset_table = dace_fieldview_util.connectivity_identifier(offset)
        # initially, the storage for the connectivty tables is created as transient;
        # when the tables are used, the storage is changed to non-transient,
        # so the corresponding arrays are supposed to be allocated by the SDFG caller
        self.sdfg.arrays[offset_table].transient = False
        offset_table_node = self.state.add_access(offset_table)

        field_array_shape = tuple(
            shape
            for dim, shape in zip(it.dimensions, field_desc.shape, strict=True)
            if dim == offset_provider.neighbor_axis.value
        )
        assert len(field_array_shape) == 1

        # we build a nested SDFG to gather all neighbors for each point in the field domain
        # it can be seen as a library node
        nsdfg, field_name, index_name, output_name = build_neighbors_sdfg(
            field_desc.dtype,
            field_array_shape,
            (offset_provider.max_neighbors,),
            self.sdfg.arrays[offset_table].dtype,
        )

        neighbors_node = self.state.add_nested_sdfg(
            nsdfg, self.sdfg, {field_name, index_name}, {output_name}
        )

        self._add_input_connection(
            it.field,
            sbs.Range(
                [
                    (0, size - 1, 1)
                    if dim == offset_provider.neighbor_axis.value
                    else (it.indices[dim].value, it.indices[dim].value, 1)  # type: ignore[union-attr]
                    for dim, size in zip(it.dimensions, field_desc.shape, strict=True)
                ]
            ),
            neighbors_node,
            field_name,
        )

        self._add_input_connection(
            offset_table_node,
            sbs.Range(
                [
                    (origin_index.value, origin_index.value, 1),
                    (0, offset_provider.max_neighbors - 1, 1),
                ]
            ),
            neighbors_node,
            index_name,
        )

        return self._get_tasklet_result(
            field_desc.dtype,
            neighbors_node,
            output_name,
            subset=sbs.Range([(0, offset_provider.max_neighbors - 1, 1)]),
        )

    def _visit_reduce(self, node: itir.FunCall) -> ValueExpr:
        assert isinstance(node.fun, itir.FunCall)
        assert len(node.fun.args) == 2
        op_name = node.fun.args[0]
        assert isinstance(op_name, itir.SymRef)
        reduce_identity = node.fun.args[1]
        assert isinstance(reduce_identity, itir.Literal)

        assert len(node.args) == 1
        input_expr = self.visit(node.args[0])
        assert isinstance(input_expr, MemletExpr | ValueExpr)
        input_desc = input_expr.node.desc(self.sdfg)

        assert isinstance(input_desc, dace.data.Array)
        if len(input_desc.shape) > 1:
            ndims = len(input_desc.shape) - 1
            reduce_axes = [ndims]
        else:
            ndims = 0
            reduce_axes = None

        reduce_wcr = "lambda x, y: " + MATH_BUILTINS_MAPPING[str(op_name)].format("x", "y")
        reduce_node = self.state.add_reduce(reduce_wcr, reduce_axes, reduce_identity)

        input_subset = sbs.Range([(0, dim_size - 1, 1) for dim_size in input_desc.shape])
        if ndims > 0:
            result_subset = sbs.Range(input_subset[0:ndims])
        else:
            result_subset = None

        if isinstance(input_expr, MemletExpr):
            self._add_input_connection(input_expr.node, input_subset, reduce_node)
        else:
            self.state.add_nedge(
                input_expr.node,
                reduce_node,
                dace.Memlet(data=input_expr.node.data, subset=input_subset),
            )

        # TODO: use type inference to determine the result type
        return self._get_tasklet_result(input_desc.dtype, reduce_node, None, result_subset)

    def _split_shift_args(
        self, args: list[itir.Expr]
    ) -> tuple[list[itir.Expr], Optional[list[itir.Expr]]]:
        """
        Splits the arguments to `shift` builtin function as pairs, each pair containing
        the offset provider and the offset value in one dimension.
        """
        pairs = [args[i : i + 2] for i in range(0, len(args), 2)]
        assert len(pairs) >= 1
        assert all(len(pair) == 2 for pair in pairs)
        return pairs[-1], list(itertools.chain(*pairs[0:-1])) if len(pairs) > 1 else None

    def _make_shift_for_rest(self, rest: list[itir.Expr], iterator: itir.Expr) -> itir.FunCall:
        """Transforms a multi-dimensional shift into recursive shift calls, each in a single dimension."""
        return itir.FunCall(
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest),
            args=[iterator],
        )

    def _make_cartesian_shift(
        self, it: IteratorExpr, offset_dim: Dimension, offset_expr: IteratorIndexExpr
    ) -> IteratorExpr:
        """Implements cartesian shift along one dimension."""
        assert offset_dim.value in it.dimensions
        new_index: SymbolExpr | ValueExpr
        assert offset_dim.value in it.indices
        index_expr = it.indices[offset_dim.value]
        if isinstance(index_expr, SymbolExpr) and isinstance(offset_expr, SymbolExpr):
            # purely symbolic expression which can be interpreted at compile time
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
                    if input_connector == "index":
                        dtype = input_expr.node.desc(self.sdfg).dtype
                    self._add_input_connection(
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
            {
                dim: (new_index if dim == offset_dim.value else index)
                for dim, index in it.indices.items()
            },
        )

    def _make_dynamic_neighbor_offset(
        self,
        offset_expr: MemletExpr | ValueExpr,
        offset_table_node: dace.nodes.AccessNode,
        origin_index: SymbolExpr,
    ) -> ValueExpr:
        """
        Implements access to neighbor connectivity table by means of a tasklet node.

        It requires a dynamic offset value, either obtained from a field (`MemletExpr`)
        or computed byanother tasklet (`ValueExpr`).
        """
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
        connectivity: Connectivity,
        offset_table_node: dace.nodes.AccessNode,
        offset_expr: IteratorIndexExpr,
    ) -> IteratorExpr:
        """Implements shift in unstructured domain by means of a neighbor table."""
        neighbor_dim = connectivity.neighbor_axis.value
        assert neighbor_dim in it.dimensions
        assert neighbor_dim not in it.indices

        origin_dim = connectivity.origin_axis.value
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

        return IteratorExpr(it.field, it.dimensions, shifted_indices)

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr:
        shift_node = node.fun
        assert isinstance(shift_node, itir.FunCall)

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
            assert isinstance(dynamic_offset_expr, MemletExpr | ValueExpr)
            offset_expr = dynamic_offset_expr

        if isinstance(offset_provider, Dimension):
            return self._make_cartesian_shift(it, offset_provider, offset_expr)
        else:
            # initially, the storage for the connectivty tables is created as transient;
            # when the tables are used, the storage is changed to non-transient,
            # so the corresponding arrays are supposed to be allocated by the SDFG caller
            offset_table = dace_fieldview_util.connectivity_identifier(offset)
            self.sdfg.arrays[offset_table].transient = False
            offset_table_node = self.state.add_access(offset_table)

            return self._make_unstructured_shift(
                it, offset_provider, offset_table_node, offset_expr
            )

    def visit_FunCall(self, node: itir.FunCall) -> IteratorExpr | MemletExpr | ValueExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif cpm.is_call_to(node, "neighbors"):
            return self._visit_neighbors(node)

        elif cpm.is_applied_reduce(node):
            return self._visit_reduce(node)

        elif cpm.is_applied_shift(node.fun):
            return self._visit_shift(node)

        else:
            assert isinstance(node.fun, itir.SymRef)

        # create a tasklet node implementing the builtin function
        builtin_name = str(node.fun.id)
        if builtin_name in MATH_BUILTINS_MAPPING:
            fmt = MATH_BUILTINS_MAPPING[builtin_name]
        else:
            raise NotImplementedError(f"'{builtin_name}' not implemented.")

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
        code = fmt.format(*node_internals)

        out_connector = "result"
        tasklet_node = self.state.add_tasklet(
            builtin_name,
            node_connections.keys(),
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
                self._add_input_connection(arg_expr.node, arg_expr.subset, tasklet_node, connector)

        # TODO: use type inference to determine the result type
        if len(node_connections) == 1 and isinstance(node_connections["__inp_0"], MemletExpr):
            dtype = node_connections["__inp_0"].node.desc(self.sdfg).dtype
        else:
            node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            dtype = dace_fieldview_util.as_dace_type(node_type)

        return self._get_tasklet_result(dtype, tasklet_node, "result")

    def visit_Lambda(
        self, node: itir.Lambda, args: list[IteratorExpr | MemletExpr | SymbolExpr]
    ) -> tuple[list[InputConnection], ValueExpr]:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr: MemletExpr | SymbolExpr | ValueExpr = self.visit(node.expr)
        if isinstance(output_expr, ValueExpr):
            return self.input_connections, output_expr

        if isinstance(output_expr, MemletExpr):
            # special case where the field operator is simply copying data from source to destination node
            output_dtype = output_expr.node.desc(self.sdfg).dtype
            tasklet_node = self.state.add_tasklet("copy", {"__inp"}, {"__out"}, "__out = __inp")
            self._add_input_connection(output_expr.node, output_expr.subset, tasklet_node, "__inp")
        else:
            # even simpler case, where a constant value is written to destination node
            output_dtype = output_expr.dtype
            tasklet_node = self.state.add_tasklet(
                "write", {}, {"__out"}, f"__out = {output_expr.value}"
            )
        return self.input_connections, self._get_tasklet_result(output_dtype, tasklet_node, "__out")

    def visit_Literal(self, node: itir.Literal) -> SymbolExpr:
        dtype = dace_fieldview_util.as_dace_type(node.type)
        return SymbolExpr(node.value, dtype)

    def visit_SymRef(self, node: itir.SymRef) -> IteratorExpr | MemletExpr | SymbolExpr:
        param = str(node.id)
        assert param in self.symbol_map
        return self.symbol_map[param]


class PythonCodegen(codegen.TemplatedGenerator):
    """Helper class to visit a symbolic expression and translate it to Python code.

    The generated Python code can be use either as the body of a tasklet node or,
    as in the case of field domain definitions, for sybolic array shape and map range.
    """

    SymRef = as_fmt("{id}")
    Literal = as_fmt("{value}")

    def _visit_deref(self, node: itir.FunCall) -> str:
        assert len(node.args) == 1
        if isinstance(node.args[0], itir.SymRef):
            return self.visit(node.args[0])
        raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")

    def _visit_numeric_builtin(self, node: itir.FunCall) -> str:
        assert isinstance(node.fun, itir.SymRef)
        fmt = MATH_BUILTINS_MAPPING[str(node.fun.id)]
        args = self.visit(node.args)
        return fmt.format(*args)

    def visit_FunCall(self, node: itir.FunCall) -> str:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)
        elif isinstance(node.fun, itir.SymRef):
            builtin_name = str(node.fun.id)
            if builtin_name in MATH_BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            else:
                raise NotImplementedError(f"'{builtin_name}' not implemented.")
        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")
