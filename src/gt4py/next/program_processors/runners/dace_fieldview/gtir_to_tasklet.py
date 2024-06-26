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
from gt4py.next.common import _DEFAULT_SKIP_VALUE as neighbor_skip_value, Connectivity, Dimension
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
    field_type: ts.FieldType | ts.ScalarType


IteratorIndexExpr: TypeAlias = MemletExpr | SymbolExpr | ValueExpr


@dataclass(frozen=True)
class IteratorExpr:
    """Iterator for field access to be consumed by `deref` or `shift` builtin functions."""

    field: dace.nodes.AccessNode
    dimensions: list[Dimension]
    indices: dict[Dimension, IteratorIndexExpr]


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


def get_reduce_params(node: itir.FunCall) -> tuple[str, SymbolExpr, SymbolExpr]:
    # TODO: use type inference to determine the result type
    dtype = dace.float64

    assert isinstance(node.fun, itir.FunCall)
    assert len(node.fun.args) == 2
    assert isinstance(node.fun.args[0], itir.SymRef)
    op_name = str(node.fun.args[0])
    assert isinstance(node.fun.args[1], itir.Literal)
    reduce_init = SymbolExpr(node.fun.args[1].value, dtype)

    if op_name not in DACE_REDUCTION_MAPPING:
        raise RuntimeError(f"Reduction operation '{op_name}' not supported.")
    identity_value = dace.dtypes.reduction_identity(dtype, DACE_REDUCTION_MAPPING[op_name])
    reduce_identity = SymbolExpr(identity_value, dtype)

    return op_name, reduce_init, reduce_identity


class LambdaToTasklet(eve.NodeVisitor):
    """Translates an `ir.Lambda` expression to a dataflow graph.

    Lambda functions should only be encountered as argument to the `as_field_op`
    builtin function, therefore the dataflow graph generated here typically
    represents the stencil function of a field operator.
    """

    sdfg: dace.SDFG
    state: dace.SDFGState
    offset_provider: dict[str, Connectivity | Dimension]
    symbol_map: dict[str, IteratorExpr | MemletExpr | SymbolExpr]
    reduce_identity: Optional[SymbolExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        map_entry: dace.nodes.MapEntry,
        offset_provider: dict[str, Connectivity | Dimension],
        reduce_identity: Optional[SymbolExpr],
    ):
        self.sdfg = sdfg
        self.state = state
        self.map_entry = map_entry
        self.offset_provider = offset_provider
        self.symbol_map = {}
        self.reduce_identity = reduce_identity

    def _add_entry_memlet_path(
        self,
        *path_nodes: dace.nodes.Node,
        memlet: Optional[dace.Memlet] = None,
        src_conn: Optional[str] = None,
        dst_conn: Optional[str] = None,
    ) -> None:
        self.state.add_memlet_path(
            path_nodes[0],
            self.map_entry,
            *path_nodes[1:],
            memlet=memlet,
            src_conn=src_conn,
            dst_conn=dst_conn,
        )

    def _get_tasklet_result(
        self,
        dtype: dace.typeclass,
        src_node: dace.nodes.Node,
        src_connector: Optional[str] = None,
        offset: Optional[str] = None,
    ) -> ValueExpr:
        data_type: ts.FieldType | ts.ScalarType
        if offset:
            offset_provider = self.offset_provider[offset]
            assert isinstance(offset_provider, Connectivity)
            var_name, _ = self.sdfg.add_array(
                "var", (offset_provider.max_neighbors,), dtype, transient=True, find_new_name=True
            )
            var_subset = f"0:{offset_provider.max_neighbors}"
            data_type = dace_fieldview_util.get_neighbors_field_type(offset, dtype)
        else:
            var_name, _ = self.sdfg.add_scalar("var", dtype, transient=True, find_new_name=True)
            var_subset = "0"
            data_type = dace_fieldview_util.as_scalar_type(str(dtype.as_numpy_dtype()))
        var_node = self.state.add_access(var_name)
        self.state.add_edge(
            src_node,
            src_connector,
            var_node,
            None,
            dace.Memlet(data=var_node.data, subset=var_subset),
        )
        return ValueExpr(var_node, data_type)

    def _visit_deref(self, node: itir.FunCall) -> MemletExpr | ValueExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, IteratorExpr):
            field_desc = it.field.desc(self.sdfg)
            assert len(field_desc.shape) == len(it.dimensions)
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # when all indices are symblic expressions, we can perform direct field access through a memlet
                field_subset = sbs.Range(
                    [
                        (it.indices[dim].value, it.indices[dim].value, 1)  # type: ignore[union-attr]
                        if dim in it.indices
                        else (0, size - 1, 1)
                        for dim, size in zip(it.dimensions, field_desc.shape)
                    ]
                )
                return MemletExpr(it.field, field_subset)

            else:
                # we use a tasklet to perform dereferencing of a generic iterator
                assert all(dim in it.indices for dim in it.dimensions)
                field_indices = [(dim, it.indices[dim]) for dim in it.dimensions]
                index_connectors = [
                    INDEX_CONNECTOR_FMT.format(dim=dim.value)
                    for dim, index in field_indices
                    if not isinstance(index, SymbolExpr)
                ]
                index_internals = ",".join(
                    str(index.value)
                    if isinstance(index, SymbolExpr)
                    else INDEX_CONNECTOR_FMT.format(dim=dim.value)
                    for dim, index in field_indices
                )
                deref_node = self.state.add_tasklet(
                    "deref_field_indirection",
                    {"field"} | set(index_connectors),
                    {"val"},
                    code=f"val = field[{index_internals}]",
                )
                # add new termination point for this field parameter
                self._add_entry_memlet_path(
                    it.field,
                    deref_node,
                    dst_conn="field",
                    memlet=dace.Memlet.from_array(it.field.data, field_desc),
                )

                for dim, index_expr in field_indices:
                    deref_connector = INDEX_CONNECTOR_FMT.format(dim=dim.value)
                    if isinstance(index_expr, MemletExpr):
                        self._add_entry_memlet_path(
                            index_expr.node,
                            deref_node,
                            dst_conn=deref_connector,
                            memlet=dace.Memlet(data=index_expr.node.data, subset=index_expr.subset),
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
        assert offset_provider.neighbor_axis in it.dimensions
        assert offset_provider.neighbor_axis not in it.indices
        assert offset_provider.origin_axis not in it.dimensions
        assert offset_provider.origin_axis in it.indices
        origin_index = it.indices[offset_provider.origin_axis]
        assert isinstance(origin_index, SymbolExpr)
        assert all(isinstance(index, SymbolExpr) for index in it.indices.values())

        field_desc = it.field.desc(self.sdfg)
        connectivity = dace_fieldview_util.connectivity_identifier(offset)
        # initially, the storage for the connectivty tables is created as transient;
        # when the tables are used, the storage is changed to non-transient,
        # so the corresponding arrays are supposed to be allocated by the SDFG caller
        connectivity_desc = self.sdfg.arrays[connectivity]
        connectivity_desc.transient = False
        connectivity_node = self.state.add_access(connectivity)

        me, mx = self.state.add_map(
            "neighbors",
            dict(__neighbor_idx=f"0:{offset_provider.max_neighbors}"),
        )
        index_connector = "__index"
        if offset_provider.has_skip_values:
            assert self.reduce_identity is not None
            skip_value_code = f" if {index_connector} != {neighbor_skip_value} else {self.reduce_identity.dtype}({self.reduce_identity.value})"
        else:
            skip_value_code = ""
        index_internals = ",".join(
            [
                it.indices[dim].value if dim != offset_provider.neighbor_axis else index_connector  # type: ignore[union-attr]
                for dim in it.dimensions
            ]
        )
        tasklet_node = self.state.add_tasklet(
            "gather_neighbors",
            {"__field", index_connector},
            {"__val"},
            f"__val = __field[{index_internals}]" + skip_value_code,
        )
        self._add_entry_memlet_path(
            it.field,
            me,
            tasklet_node,
            dst_conn="__field",
            memlet=dace.Memlet.from_array(it.field.data, field_desc),
        )
        self._add_entry_memlet_path(
            connectivity_node,
            me,
            tasklet_node,
            dst_conn=index_connector,
            memlet=dace.Memlet(
                data=connectivity, subset=sbs.Indices([origin_index.value, "__neighbor_idx"])
            ),
        )
        neighbor_val_name, neighbor_val_array = self.sdfg.add_array(
            "neighbor_val",
            (offset_provider.max_neighbors,),
            field_desc.dtype,
            transient=True,
            find_new_name=True,
        )
        neighbor_val_node = self.state.add_access(neighbor_val_name)
        self.state.add_memlet_path(
            tasklet_node,
            mx,
            neighbor_val_node,
            src_conn="__val",
            memlet=dace.Memlet(data=neighbor_val_name, subset="__neighbor_idx"),
        )
        neighbors_field_type = dace_fieldview_util.get_neighbors_field_type(
            offset, field_desc.dtype
        )
        return ValueExpr(neighbor_val_node, neighbors_field_type)

    def _visit_reduce(self, node: itir.FunCall) -> ValueExpr:
        op_name, reduce_init, reduce_identity = get_reduce_params(node)
        dtype = reduce_identity.dtype

        # we store the value of reduce identity in the visitor context while visiting the input to reduction
        # this value will be returned by the neighbors builtin function for skip values
        prev_reduce_identity = self.reduce_identity
        self.reduce_identity = reduce_identity
        assert len(node.args) == 1
        input_expr = self.visit(node.args[0])
        assert isinstance(input_expr, MemletExpr | ValueExpr)
        self.reduce_identity = prev_reduce_identity
        input_desc = input_expr.node.desc(self.sdfg)
        assert isinstance(input_desc, dace.data.Array)

        if len(input_desc.shape) > 1:
            assert isinstance(input_expr, MemletExpr)
            ndims = len(input_desc.shape) - 1
            assert set(input_expr.subset.size()[0:ndims]) == {1}
            reduce_axes = [ndims]
        else:
            reduce_axes = None
        res_var, res_desc = self.sdfg.add_scalar("var", dtype, find_new_name=True, transient=True)
        res_type = dace_fieldview_util.as_scalar_type(str(dtype.as_numpy_dtype()))

        reduce_wcr = "lambda x, y: " + MATH_BUILTINS_MAPPING[op_name].format("x", "y")
        reduce_node = self.state.add_reduce(reduce_wcr, reduce_axes, reduce_init.value)

        if isinstance(input_expr, MemletExpr):
            self._add_entry_memlet_path(
                input_expr.node,
                reduce_node,
                memlet=dace.Memlet(data=input_expr.node.data, subset=input_expr.subset),
            )
        else:
            self.state.add_nedge(
                input_expr.node,
                reduce_node,
                dace.Memlet.from_array(input_expr.node.data, input_desc),
            )

        res_node = self.state.add_access(res_var)
        self.state.add_nedge(
            reduce_node,
            res_node,
            dace.Memlet.from_array(res_var, res_desc),
        )
        return ValueExpr(res_node, res_type)

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
        assert offset_dim in it.dimensions
        new_index: SymbolExpr | ValueExpr
        assert offset_dim in it.indices
        index_expr = it.indices[offset_dim]
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
                    self._add_entry_memlet_path(
                        input_expr.node,
                        dynamic_offset_tasklet,
                        dst_conn=input_connector,
                        memlet=dace.Memlet(data=input_expr.node.data, subset=input_expr.subset),
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
        self._add_entry_memlet_path(
            offset_table_node,
            tasklet_node,
            dst_conn="table",
            memlet=dace.Memlet.from_array(
                offset_table_node.data, offset_table_node.desc(self.sdfg)
            ),
        )
        if isinstance(offset_expr, MemletExpr):
            self._add_entry_memlet_path(
                offset_expr.node,
                tasklet_node,
                dst_conn="offset",
                memlet=dace.Memlet(data=offset_expr.node.data, subset=offset_expr.subset),
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

        elif cpm.is_applied_shift(node):
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
                self._add_entry_memlet_path(
                    arg_expr.node,
                    tasklet_node,
                    dst_conn=connector,
                    memlet=dace.Memlet(data=arg_expr.node.data, subset=arg_expr.subset),
                )

        # TODO: use type inference to determine the result type
        if len(node_connections) == 1 and isinstance(node_connections["__inp_0"], MemletExpr):
            dtype = node_connections["__inp_0"].node.desc(self.sdfg).dtype
        else:
            node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
            dtype = dace_fieldview_util.as_dace_type(node_type)

        return self._get_tasklet_result(dtype, tasklet_node, "result")

    def visit_Lambda(
        self, node: itir.Lambda, args: list[IteratorExpr | MemletExpr | SymbolExpr]
    ) -> ValueExpr:
        for p, arg in zip(node.params, args, strict=True):
            self.symbol_map[str(p.id)] = arg
        output_expr: MemletExpr | SymbolExpr | ValueExpr = self.visit(node.expr)
        if isinstance(output_expr, ValueExpr):
            return output_expr

        if isinstance(output_expr, MemletExpr):
            # special case where the field operator is simply copying data from source to destination node
            dtype = self.sdfg.arrays[output_expr.node.data].dtype
            scalar_type = dace_fieldview_util.as_scalar_type(str(dtype.as_numpy_dtype()))
            var, _ = self.sdfg.add_scalar("var", dtype, find_new_name=True, transient=True)
            result_node = self.state.add_access(var)
            self._add_entry_memlet_path(
                output_expr.node,
                result_node,
                memlet=dace.Memlet(data=output_expr.node.data, subset=output_expr.subset),
            )
            return ValueExpr(result_node, scalar_type)
        else:
            # even simpler case, where a constant value is written to destination node
            output_dtype = output_expr.dtype
            tasklet_node = self.state.add_tasklet(
                "write", {}, {"__out"}, f"__out = {output_expr.value}"
            )
            return self._get_tasklet_result(output_dtype, tasklet_node, "__out")

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
