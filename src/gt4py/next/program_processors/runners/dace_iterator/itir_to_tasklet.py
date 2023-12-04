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
import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeAlias, cast

import dace
import numpy as np
from dace import subsets
from dace.transformation.dataflow import MapFusion
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols

import gt4py.eve.codegen
from gt4py import eve
from gt4py.next import Dimension, StridedNeighborOffsetProvider, type_inference as next_typing
from gt4py.next.iterator import ir as itir, type_inference as itir_typing
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.iterator.ir import FunCall, Lambda
from gt4py.next.iterator.type_inference import Val
from gt4py.next.type_system import type_specifications as ts

from .utility import (
    add_mapped_nested_sdfg,
    as_dace_type,
    connectivity_identifier,
    create_memlet_at,
    create_memlet_full,
    filter_neighbor_tables,
    flatten_list,
    map_nested_sdfg_symbols,
    new_array_symbols,
    unique_name,
    unique_var_name,
)


_TYPE_MAPPING = {
    "float": dace.float64,
    "float32": dace.float32,
    "float64": dace.float64,
    "int": dace.int32 if np.dtype(int).itemsize == 4 else dace.int64,
    "int32": dace.int32,
    "int64": dace.int64,
    "bool": dace.bool_,
}


def itir_type_as_dace_type(type_: next_typing.Type):
    if isinstance(type_, itir_typing.Primitive):
        return _TYPE_MAPPING[type_.name]
    raise NotImplementedError()


def get_reduce_identity_value(op_name_: str, type_: Any):
    if op_name_ == "plus":
        init_value = type_(0)
    elif op_name_ == "multiplies":
        init_value = type_(1)
    elif op_name_ == "minimum":
        init_value = type_("inf")
    elif op_name_ == "maximum":
        init_value = type_("-inf")
    else:
        raise NotImplementedError()

    return init_value


_MATH_BUILTINS_MAPPING = {
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


# Define type of variables used for field indexing
_INDEX_DTYPE = _TYPE_MAPPING["int64"]


@dataclasses.dataclass
class SymbolExpr:
    value: dace.symbolic.SymbolicType
    dtype: dace.typeclass


@dataclasses.dataclass
class ValueExpr:
    value: dace.nodes.AccessNode
    dtype: dace.typeclass


@dataclasses.dataclass
class IteratorExpr:
    field: dace.nodes.AccessNode
    indices: dict[str, dace.nodes.AccessNode]
    dtype: dace.typeclass
    dimensions: list[str]


# Union of possible expression types
TaskletExpr: TypeAlias = IteratorExpr | SymbolExpr | ValueExpr


@dataclasses.dataclass
class Context:
    body: dace.SDFG
    state: dace.SDFGState
    symbol_map: dict[str, TaskletExpr]
    # if we encounter a reduction node, the reduction state needs to be pushed to child nodes
    reduce_limit: int
    reduce_wcr: Optional[str]

    def __init__(
        self,
        body: dace.SDFG,
        state: dace.SDFGState,
        symbol_map: dict[str, TaskletExpr],
        reduce_limit: int = 0,
        reduce_wcr: Optional[str] = None,
    ):
        self.body = body
        self.state = state
        self.symbol_map = symbol_map
        self.reduce_limit = reduce_limit
        self.reduce_wcr = reduce_wcr


def builtin_neighbors(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    offset_literal, data = node_args
    assert isinstance(offset_literal, itir.OffsetLiteral)
    offset_dim = offset_literal.value
    assert isinstance(offset_dim, str)
    iterator = transformer.visit(data)
    table: NeighborTableOffsetProvider = transformer.offset_provider[offset_dim]
    assert isinstance(table, NeighborTableOffsetProvider)

    offset = transformer.offset_provider[offset_dim]
    if isinstance(offset, Dimension):
        raise NotImplementedError(
            "Neighbor reductions for cartesian grids not implemented in DaCe backend."
        )

    sdfg: dace.SDFG = transformer.context.body
    state: dace.SDFGState = transformer.context.state

    shifted_dim = table.origin_axis.value

    result_name = unique_var_name()
    sdfg.add_array(result_name, dtype=iterator.dtype, shape=(table.max_neighbors,), transient=True)
    result_access = state.add_access(result_name)

    table_name = connectivity_identifier(offset_dim)

    # generate unique map index name to avoid conflict with other maps inside same state
    index_name = unique_name("__neigh_idx")
    me, mx = state.add_map(
        f"{offset_dim}_neighbors_map",
        ndrange={index_name: f"0:{table.max_neighbors}"},
    )
    shift_tasklet = state.add_tasklet(
        "shift",
        code=f"__result = __table[__idx, {index_name}]",
        inputs={"__table", "__idx"},
        outputs={"__result"},
    )
    data_access_tasklet = state.add_tasklet(
        "data_access",
        code="__result = __field[__idx]",
        inputs={"__field", "__idx"},
        outputs={"__result"},
    )
    idx_name = unique_var_name()
    sdfg.add_scalar(idx_name, _INDEX_DTYPE, transient=True)
    state.add_memlet_path(
        state.add_access(table_name),
        me,
        shift_tasklet,
        memlet=create_memlet_full(table_name, sdfg.arrays[table_name]),
        dst_conn="__table",
    )
    state.add_memlet_path(
        iterator.indices[shifted_dim],
        me,
        shift_tasklet,
        memlet=dace.Memlet.simple(iterator.indices[shifted_dim].data, "0"),
        dst_conn="__idx",
    )
    state.add_edge(
        shift_tasklet,
        "__result",
        data_access_tasklet,
        "__idx",
        dace.Memlet.simple(idx_name, "0"),
    )
    # select full shape only in the neighbor-axis dimension
    field_subset = tuple(
        f"0:{shape}" if dim == table.neighbor_axis.value else f"i_{dim}"
        for dim, shape in zip(sorted(iterator.dimensions), sdfg.arrays[iterator.field.data].shape)
    )
    state.add_memlet_path(
        iterator.field,
        me,
        data_access_tasklet,
        memlet=create_memlet_at(iterator.field.data, field_subset),
        dst_conn="__field",
    )
    state.add_memlet_path(
        data_access_tasklet,
        mx,
        result_access,
        memlet=dace.Memlet.simple(result_name, index_name),
        src_conn="__result",
    )

    return [ValueExpr(result_access, iterator.dtype)]


def builtin_can_deref(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    # first visit shift, to get set of indices for deref
    can_deref_callable = node_args[0]
    assert isinstance(can_deref_callable, itir.FunCall)
    shift_callable = can_deref_callable.fun
    assert isinstance(shift_callable, itir.FunCall)
    assert isinstance(shift_callable.fun, itir.SymRef)
    assert shift_callable.fun.id == "shift"
    iterator = transformer._visit_shift(can_deref_callable)

    # TODO: remove this special case when ITIR reduce-unroll pass is able to catch it
    if not isinstance(iterator, IteratorExpr):
        assert len(iterator) == 1 and isinstance(iterator[0], ValueExpr)
        # We can always deref a value expression, therefore hard-code `can_deref` to True.
        # Returning a SymbolExpr would be preferable, but it requires update to type-checking.
        result_name = unique_var_name()
        transformer.context.body.add_scalar(result_name, dace.dtypes.bool, transient=True)
        result_node = transformer.context.state.add_access(result_name)
        transformer.context.state.add_edge(
            transformer.context.state.add_tasklet("can_always_deref", {}, {"_out"}, "_out = True"),
            "_out",
            result_node,
            None,
            dace.Memlet.simple(result_name, "0"),
        )
        return [ValueExpr(result_node, dace.dtypes.bool)]

    # create tasklet to check that field indices are non-negative (-1 is invalid)
    args = [ValueExpr(access_node, _INDEX_DTYPE) for access_node in iterator.indices.values()]
    internals = [f"{arg.value.data}_v" for arg in args]
    expr_code = " and ".join([f"{v} >= 0" for v in internals])

    # TODO(edopao): select-memlet could maybe allow to efficiently translate can_deref to predicative execution
    return transformer.add_expr_tasklet(
        list(zip(args, internals)), expr_code, dace.dtypes.bool, "can_deref"
    )


def builtin_if(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [arg for li in transformer.visit(node_args) for arg in li]
    expr_args = [(arg, f"{arg.value.data}_v") for arg in args if not isinstance(arg, SymbolExpr)]
    internals = [
        arg.value if isinstance(arg, SymbolExpr) else f"{arg.value.data}_v" for arg in args
    ]
    expr = "({1} if {0} else {2})".format(*internals)
    node_type = transformer.node_types[id(node)]
    assert isinstance(node_type, itir_typing.Val)
    type_ = itir_type_as_dace_type(node_type.dtype)
    return transformer.add_expr_tasklet(expr_args, expr, type_, "if")


def builtin_list_get(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = list(itertools.chain(*transformer.visit(node_args)))
    assert len(args) == 2
    # index node
    assert isinstance(args[0], (SymbolExpr, ValueExpr))
    # 1D-array node
    assert isinstance(args[1], ValueExpr)
    # source node should be a 1D array
    assert len(transformer.context.body.arrays[args[1].value.data].shape) == 1

    expr_args = [(arg, f"{arg.value.data}_v") for arg in args if not isinstance(arg, SymbolExpr)]
    internals = [
        arg.value if isinstance(arg, SymbolExpr) else f"{arg.value.data}_v" for arg in args
    ]
    expr = f"{internals[1]}[{internals[0]}]"
    return transformer.add_expr_tasklet(expr_args, expr, args[1].dtype, "list_get")


def builtin_cast(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [transformer.visit(node_args[0])[0]]
    internals = [f"{arg.value.data}_v" for arg in args]
    target_type = node_args[1]
    assert isinstance(target_type, itir.SymRef)
    expr = _MATH_BUILTINS_MAPPING[target_type.id].format(*internals)
    node_type = transformer.node_types[id(node)]
    assert isinstance(node_type, itir_typing.Val)
    type_ = itir_type_as_dace_type(node_type.dtype)
    return transformer.add_expr_tasklet(list(zip(args, internals)), expr, type_, "cast")


def builtin_make_tuple(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [transformer.visit(arg) for arg in node_args]
    return args


def builtin_tuple_get(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    elements = transformer.visit(node_args[1])
    index = node_args[0]
    if isinstance(index, itir.Literal):
        return elements[int(index.value)]
    raise ValueError("Tuple can only be subscripted with compile-time constants")


_GENERAL_BUILTIN_MAPPING: dict[
    str, Callable[["PythonTaskletCodegen", itir.Expr, list[itir.Expr]], list[ValueExpr]]
] = {
    "can_deref": builtin_can_deref,
    "cast_": builtin_cast,
    "if_": builtin_if,
    "list_get": builtin_list_get,
    "make_tuple": builtin_make_tuple,
    "neighbors": builtin_neighbors,
    "tuple_get": builtin_tuple_get,
}


class GatherLambdaSymbolsPass(eve.NodeVisitor):
    _sdfg: dace.SDFG
    _state: dace.SDFGState
    _symbol_map: dict[str, TaskletExpr]
    _parent_symbol_map: dict[str, TaskletExpr]

    def __init__(
        self,
        sdfg,
        state,
        parent_symbol_map,
    ):
        self._sdfg = sdfg
        self._state = state
        self._symbol_map = {}
        self._parent_symbol_map = parent_symbol_map

    @property
    def symbol_refs(self):
        """Dictionary of symbols referenced from the lambda expression."""
        return self._symbol_map

    def _add_symbol(self, param, arg):
        if isinstance(arg, ValueExpr):
            # create storage in lambda sdfg
            self._sdfg.add_scalar(param, dtype=arg.dtype)
            # update table of lambda symbol
            self._symbol_map[param] = ValueExpr(self._state.add_access(param), arg.dtype)
        elif isinstance(arg, IteratorExpr):
            # create storage in lambda sdfg
            ndims = len(arg.dimensions)
            shape, strides = new_array_symbols(param, ndims)
            self._sdfg.add_array(param, shape=shape, strides=strides, dtype=arg.dtype)
            index_names = {dim: f"__{param}_i_{dim}" for dim in arg.indices.keys()}
            for _, index_name in index_names.items():
                self._sdfg.add_scalar(index_name, dtype=_INDEX_DTYPE)
            # update table of lambda symbol
            field = self._state.add_access(param)
            indices = {
                dim: self._state.add_access(index_arg) for dim, index_arg in index_names.items()
            }
            self._symbol_map[param] = IteratorExpr(field, indices, arg.dtype, arg.dimensions)
        else:
            assert isinstance(arg, SymbolExpr)
            self._symbol_map[param] = arg

    def visit_SymRef(self, node: itir.SymRef):
        name = str(node.id)
        if name in self._parent_symbol_map and name not in self._symbol_map:
            arg = self._parent_symbol_map[name]
            self._add_symbol(name, arg)

    def visit_Lambda(self, node: itir.Lambda, args: Optional[Sequence[TaskletExpr]] = None):
        if args is not None:
            assert len(node.params) == len(args)
            for param, arg in zip(node.params, args):
                self._add_symbol(str(param.id), arg)
        self.visit(node.expr)


class GatherOutputSymbolsPass(eve.NodeVisitor):
    _sdfg: dace.SDFG
    _state: dace.SDFGState
    _node_types: dict[int, next_typing.Type]
    _symbol_map: dict[str, TaskletExpr]

    @property
    def symbol_refs(self):
        """Dictionary of symbols referenced from the output expression."""
        return self._symbol_map

    def __init__(
        self,
        sdfg,
        state,
        node_types,
    ):
        self._sdfg = sdfg
        self._state = state
        self._node_types = node_types
        self._symbol_map = {}

    def visit_SymRef(self, node: itir.SymRef):
        param = str(node.id)
        if param not in _GENERAL_BUILTIN_MAPPING and param not in self._symbol_map:
            node_type = self._node_types[id(node)]
            assert isinstance(node_type, Val)
            access_node = self._state.add_access(param)
            self._symbol_map[param] = ValueExpr(
                access_node, dtype=itir_type_as_dace_type(node_type.dtype)
            )


class PythonTaskletCodegen(gt4py.eve.codegen.TemplatedGenerator):
    offset_provider: dict[str, Any]
    context: Context
    node_types: dict[int, next_typing.Type]

    def __init__(
        self,
        offset_provider: dict[str, Any],
        context: Context,
        node_types: dict[int, next_typing.Type],
    ):
        self.offset_provider = offset_provider
        self.context = context
        self.node_types = node_types

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition, **kwargs):
        raise NotImplementedError()

    def visit_Lambda(
        self, node: itir.Lambda, args: Sequence[TaskletExpr]
    ) -> tuple[
        Context,
        list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]],
        list[ValueExpr],
    ]:
        func_name = f"lambda_{abs(hash(node)):x}"
        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        connectivity_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

        # Create the SDFG for the lambda's body
        lambda_sdfg = dace.SDFG(func_name)
        lambda_state = lambda_sdfg.add_state(f"{func_name}_entry", True)

        lambda_symbols_pass = GatherLambdaSymbolsPass(
            lambda_sdfg, lambda_state, self.context.symbol_map
        )
        lambda_symbols_pass.visit(node, args=args)

        # Add for input nodes for lambda symbols
        inputs: list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]] = []
        for sym, input_node in lambda_symbols_pass.symbol_refs.items():
            arg = next((arg for param, arg in zip(node.params, args) if param.id == sym), None)
            if arg:
                outer_node = arg
            else:
                # the symbol is not found among lambda arguments, then it is inherited from parent scope
                outer_node = self.context.symbol_map[sym]
            if isinstance(input_node, IteratorExpr):
                assert isinstance(outer_node, IteratorExpr)
                index_params = {
                    dim: index_node.data for dim, index_node in input_node.indices.items()
                }
                inputs.append(((sym, index_params), outer_node))
            elif isinstance(input_node, ValueExpr):
                assert isinstance(outer_node, ValueExpr)
                inputs.append((sym, outer_node))

        # Add connectivities as arrays
        for name in connectivity_names:
            shape, strides = new_array_symbols(name, ndim=2)
            dtype = self.context.body.arrays[name].dtype
            lambda_sdfg.add_array(name, shape=shape, strides=strides, dtype=dtype)

        # Translate the lambda's body in its own context
        lambda_context = Context(
            lambda_sdfg,
            lambda_state,
            lambda_symbols_pass.symbol_refs,
            reduce_limit=self.context.reduce_limit,
            reduce_wcr=self.context.reduce_wcr,
        )
        lambda_taskgen = PythonTaskletCodegen(self.offset_provider, lambda_context, self.node_types)

        results: list[ValueExpr] = []
        # We are flattening the returned list of value expressions because the multiple outputs of a lamda
        # should be a list of nodes without tuple structure. Ideally, an ITIR transformation could do this.
        for expr in flatten_list(lambda_taskgen.visit(node.expr)):
            if isinstance(expr, ValueExpr):
                result_name = unique_var_name()
                lambda_sdfg.add_scalar(result_name, expr.dtype, transient=True)
                result_access = lambda_state.add_access(result_name)
                lambda_state.add_nedge(
                    expr.value,
                    result_access,
                    # in case of reduction lambda, the output edge from lambda tasklet performs write-conflict resolution
                    dace.Memlet.simple(result_access.data, "0", wcr_str=self.context.reduce_wcr),
                )
                result = ValueExpr(value=result_access, dtype=expr.dtype)
            else:
                # Forwarding result through a tasklet needed because empty SDFG states don't properly forward connectors
                result = lambda_taskgen.add_expr_tasklet([], expr.value, expr.dtype, "forward")[0]
            lambda_sdfg.arrays[result.value.data].transient = False
            results.append(result)

        # remove isolated access nodes for connectivity arrays not consumed by lambda
        for sub_node in lambda_state.nodes():
            if isinstance(sub_node, dace.nodes.AccessNode):
                if lambda_state.out_degree(sub_node) == 0 and lambda_state.in_degree(sub_node) == 0:
                    lambda_state.remove_node(sub_node)

        return lambda_context, inputs, results

    def visit_SymRef(self, node: itir.SymRef) -> list[ValueExpr | SymbolExpr] | IteratorExpr:
        param = str(node.id)
        value = self.context.symbol_map[param]
        if isinstance(value, (ValueExpr, SymbolExpr)):
            return [value]
        return value

    def visit_Literal(self, node: itir.Literal) -> list[SymbolExpr]:
        node_type = self.node_types[id(node)]
        assert isinstance(node_type, Val)
        return [SymbolExpr(node.value, itir_type_as_dace_type(node_type.dtype))]

    def visit_FunCall(self, node: itir.FunCall) -> list[ValueExpr] | IteratorExpr:
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        if isinstance(node.fun, itir.FunCall) and isinstance(node.fun.fun, itir.SymRef):
            if node.fun.fun.id == "shift":
                return self._visit_shift(node)
            elif node.fun.fun.id == "reduce":
                return self._visit_reduce(node)

        if isinstance(node.fun, itir.SymRef):
            builtin_name = str(node.fun.id)
            if builtin_name in _MATH_BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            elif builtin_name in _GENERAL_BUILTIN_MAPPING:
                return self._visit_general_builtin(node)
            else:
                raise NotImplementedError(f"{builtin_name} not implemented")
        return self._visit_call(node)

    def _visit_call(self, node: itir.FunCall):
        args = self.visit(node.args)
        args = [arg if isinstance(arg, Sequence) else [arg] for arg in args]
        args = list(itertools.chain(*args))

        func_context, func_inputs, results = self.visit(node.fun, args=args)

        nsdfg_inputs = {}
        for name, value in func_inputs:
            if isinstance(value, ValueExpr):
                nsdfg_inputs[name] = create_memlet_full(
                    value.value.data, self.context.body.arrays[value.value.data]
                )
            else:
                assert isinstance(value, IteratorExpr)
                field = name[0]
                indices = name[1]
                nsdfg_inputs[field] = create_memlet_full(
                    value.field.data, self.context.body.arrays[value.field.data]
                )
                for dim, var in indices.items():
                    store = value.indices[dim].data
                    nsdfg_inputs[var] = create_memlet_full(store, self.context.body.arrays[store])

        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        for conn, _ in neighbor_tables:
            var = connectivity_identifier(conn)
            nsdfg_inputs[var] = create_memlet_full(var, self.context.body.arrays[var])

        symbol_mapping = map_nested_sdfg_symbols(self.context.body, func_context.body, nsdfg_inputs)

        nsdfg_node = self.context.state.add_nested_sdfg(
            func_context.body,
            None,
            inputs=set(nsdfg_inputs.keys()),
            outputs=set(r.value.data for r in results),
            symbol_mapping=symbol_mapping,
        )

        for name, value in func_inputs:
            if isinstance(value, ValueExpr):
                value_memlet = nsdfg_inputs[name]
                self.context.state.add_edge(value.value, None, nsdfg_node, name, value_memlet)
            else:
                assert isinstance(value, IteratorExpr)
                field = name[0]
                indices = name[1]
                field_memlet = nsdfg_inputs[field]
                self.context.state.add_edge(value.field, None, nsdfg_node, field, field_memlet)
                for dim, var in indices.items():
                    store = value.indices[dim]
                    idx_memlet = nsdfg_inputs[var]
                    self.context.state.add_edge(store, None, nsdfg_node, var, idx_memlet)
        for conn, _ in neighbor_tables:
            var = connectivity_identifier(conn)
            memlet = nsdfg_inputs[var]
            access = self.context.state.add_access(var)
            self.context.state.add_edge(access, None, nsdfg_node, var, memlet)

        result_exprs = []
        for result in results:
            name = unique_var_name()
            self.context.body.add_scalar(name, result.dtype, transient=True)
            result_access = self.context.state.add_access(name)
            result_exprs.append(ValueExpr(result_access, result.dtype))
            memlet = create_memlet_full(name, self.context.body.arrays[name])
            self.context.state.add_edge(nsdfg_node, result.value.data, result_access, None, memlet)

        return result_exprs

    def _visit_deref(self, node: itir.FunCall) -> list[ValueExpr]:
        iterator = self.visit(node.args[0])
        if not isinstance(iterator, IteratorExpr):
            # already a list of ValueExpr
            return iterator

        args: list[ValueExpr]
        sorted_dims = sorted(iterator.dimensions)
        if self.context.reduce_limit:
            # we are visiting a child node of reduction, so the neighbor index can be used for indirect addressing
            result_name = unique_var_name()
            self.context.body.add_array(
                result_name,
                dtype=iterator.dtype,
                shape=(self.context.reduce_limit,),
                transient=True,
            )
            result_access = self.context.state.add_access(result_name)

            # generate unique map index name to avoid conflict with other maps inside same state
            index_name = unique_name("__deref_idx")
            me, mx = self.context.state.add_map(
                "deref_map",
                ndrange={index_name: f"0:{self.context.reduce_limit}"},
            )

            # if dim is not found in iterator indices, we take the neighbor index over the reduction domain
            flat_index = [
                f"{iterator.indices[dim].data}_v" if dim in iterator.indices else index_name
                for dim in sorted_dims
            ]
            args = [ValueExpr(iterator.field, iterator.dtype)] + [
                ValueExpr(iterator.indices[dim], _INDEX_DTYPE) for dim in iterator.indices
            ]
            internals = [f"{arg.value.data}_v" for arg in args]

            deref_tasklet = self.context.state.add_tasklet(
                name="deref",
                inputs=set(internals),
                outputs={"__result"},
                code=f"__result = {args[0].value.data}_v[{', '.join(flat_index)}]",
            )

            for arg, internal in zip(args, internals):
                input_memlet = create_memlet_full(
                    arg.value.data, self.context.body.arrays[arg.value.data]
                )
                self.context.state.add_memlet_path(
                    arg.value, me, deref_tasklet, memlet=input_memlet, dst_conn=internal
                )

            self.context.state.add_memlet_path(
                deref_tasklet,
                mx,
                result_access,
                memlet=dace.Memlet.simple(result_name, index_name),
                src_conn="__result",
            )

            return [ValueExpr(value=result_access, dtype=iterator.dtype)]

        elif all([dim in iterator.indices for dim in iterator.dimensions]):
            # The deref iterator has index values on all dimensions: the result will be a scalar
            args = [ValueExpr(iterator.field, iterator.dtype)] + [
                ValueExpr(iterator.indices[dim], _INDEX_DTYPE) for dim in sorted_dims
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{', '.join(internals[1:])}]"
            return self.add_expr_tasklet(list(zip(args, internals)), expr, iterator.dtype, "deref")

        else:
            # Not all dimensions are included in the deref index list:
            # this means the ND-field will be sliced along one or more dimensions and the result will be an array
            field_array = self.context.body.arrays[iterator.field.data]
            result_shape = tuple(
                dim_size
                for dim, dim_size in zip(sorted_dims, field_array.shape)
                if dim not in iterator.indices
            )
            result_name = unique_var_name()
            self.context.body.add_array(result_name, result_shape, iterator.dtype, transient=True)
            result_array = self.context.body.arrays[result_name]
            result_node = self.context.state.add_access(result_name)

            deref_connectors = ["_inp"] + [
                f"_i_{dim}" for dim in sorted_dims if dim in iterator.indices
            ]
            deref_nodes = [iterator.field] + [
                iterator.indices[dim] for dim in sorted_dims if dim in iterator.indices
            ]
            deref_memlets = [dace.Memlet.from_array(iterator.field.data, field_array)] + [
                dace.Memlet.simple(node.data, "0") for node in deref_nodes[1:]
            ]

            # we create a nested sdfg in order to access the index scalar values as symbols in a memlet subset
            deref_sdfg = dace.SDFG("deref")
            deref_sdfg.add_array(
                "_inp", field_array.shape, iterator.dtype, strides=field_array.strides
            )
            for connector in deref_connectors[1:]:
                deref_sdfg.add_scalar(connector, _INDEX_DTYPE)
            deref_sdfg.add_array("_out", result_shape, iterator.dtype)
            deref_init_state = deref_sdfg.add_state("init", True)
            deref_access_state = deref_sdfg.add_state("access")
            deref_sdfg.add_edge(
                deref_init_state,
                deref_access_state,
                dace.InterstateEdge(
                    assignments={f"_sym{inp}": inp for inp in deref_connectors[1:]}
                ),
            )
            # we access the size in source field shape as symbols set on the nested sdfg
            source_subset = tuple(
                f"_sym_i_{dim}" if dim in iterator.indices else f"0:{size}"
                for dim, size in zip(sorted_dims, field_array.shape)
            )
            deref_access_state.add_nedge(
                deref_access_state.add_access("_inp"),
                deref_access_state.add_access("_out"),
                dace.Memlet(
                    data="_out",
                    subset=subsets.Range.from_array(result_array),
                    other_subset=",".join(source_subset),
                ),
            )

            deref_node = self.context.state.add_nested_sdfg(
                deref_sdfg,
                self.context.body,
                inputs=set(deref_connectors),
                outputs={"_out"},
            )
            for connector, node, memlet in zip(deref_connectors, deref_nodes, deref_memlets):
                self.context.state.add_edge(node, None, deref_node, connector, memlet)
            self.context.state.add_edge(
                deref_node,
                "_out",
                result_node,
                None,
                dace.Memlet.from_array(result_name, result_array),
            )
            return [ValueExpr(result_node, iterator.dtype)]

    def _split_shift_args(
        self, args: list[itir.Expr]
    ) -> tuple[list[itir.Expr], Optional[list[itir.Expr]]]:
        pairs = [args[i : i + 2] for i in range(0, len(args), 2)]
        assert len(pairs) >= 1
        assert all(len(pair) == 2 for pair in pairs)
        return pairs[-1], list(itertools.chain(*pairs[0:-1])) if len(pairs) > 1 else None

    def _make_shift_for_rest(self, rest, iterator):
        return itir.FunCall(
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest), args=[iterator]
        )

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr | list[ValueExpr]:
        shift = node.fun
        assert isinstance(shift, itir.FunCall)
        tail, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0])
        if not isinstance(iterator, IteratorExpr):
            # shift cannot be applied because the argument is not iterable
            # TODO: remove this special case when ITIR reduce-unroll pass is able to catch it
            assert isinstance(iterator, list) and len(iterator) == 1
            assert isinstance(iterator[0], ValueExpr)
            return iterator

        assert isinstance(tail[0], itir.OffsetLiteral)
        offset_dim = tail[0].value
        assert isinstance(offset_dim, str)
        offset_node = self.visit(tail[1])[0]
        assert offset_node.dtype in dace.dtypes.INTEGER_TYPES

        if isinstance(self.offset_provider[offset_dim], NeighborTableOffsetProvider):
            offset_provider = self.offset_provider[offset_dim]
            connectivity = self.context.state.add_access(connectivity_identifier(offset_dim))

            shifted_dim = offset_provider.origin_axis.value
            target_dim = offset_provider.neighbor_axis.value
            args = [
                ValueExpr(connectivity, offset_provider.table.dtype),
                ValueExpr(iterator.indices[shifted_dim], offset_node.dtype),
                offset_node,
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{internals[1]}, {internals[2]}]"
        elif isinstance(self.offset_provider[offset_dim], StridedNeighborOffsetProvider):
            offset_provider = self.offset_provider[offset_dim]

            shifted_dim = offset_provider.origin_axis.value
            target_dim = offset_provider.neighbor_axis.value
            args = [
                ValueExpr(iterator.indices[shifted_dim], offset_node.dtype),
                offset_node,
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]} * {offset_provider.max_neighbors} + {internals[1]}"
        else:
            assert isinstance(self.offset_provider[offset_dim], Dimension)

            shifted_dim = self.offset_provider[offset_dim].value
            target_dim = shifted_dim
            args = [
                ValueExpr(iterator.indices[shifted_dim], offset_node.dtype),
                offset_node,
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]} + {internals[1]}"

        shifted_value = self.add_expr_tasklet(
            list(zip(args, internals)), expr, offset_node.dtype, "shift"
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        del shifted_index[shifted_dim]
        shifted_index[target_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index, iterator.dtype, iterator.dimensions)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral) -> list[ValueExpr]:
        offset = node.value
        assert isinstance(offset, int)
        offset_var = unique_var_name()
        self.context.body.add_scalar(offset_var, _INDEX_DTYPE, transient=True)
        offset_node = self.context.state.add_access(offset_var)
        tasklet_node = self.context.state.add_tasklet(
            "get_offset", {}, {"__out"}, f"__out = {offset}"
        )
        self.context.state.add_edge(
            tasklet_node, "__out", offset_node, None, dace.Memlet.simple(offset_var, "0")
        )
        return [ValueExpr(offset_node, self.context.body.arrays[offset_var].dtype)]

    def _visit_reduce(self, node: itir.FunCall):
        result_name = unique_var_name()
        result_access = self.context.state.add_access(result_name)

        if len(node.args) == 1:
            assert (
                isinstance(node.args[0], itir.FunCall)
                and isinstance(node.args[0].fun, itir.SymRef)
                and node.args[0].fun.id == "neighbors"
            )
            args = self.visit(node.args)
            assert len(args) == 1
            args = args[0]
            assert len(args) == 1
            neighbors_expr = args[0]
            result_dtype = neighbors_expr.dtype
            assert isinstance(node.fun, itir.FunCall)
            op_name = node.fun.args[0]
            assert isinstance(op_name, itir.SymRef)
            init = node.fun.args[1]

            reduce_array_desc = neighbors_expr.value.desc(self.context.body)

            self.context.body.add_scalar(result_name, result_dtype, transient=True)
            op_str = _MATH_BUILTINS_MAPPING[str(op_name)].format("__result", "__values[__idx]")
            reduce_tasklet = self.context.state.add_tasklet(
                "reduce",
                code=f"__result = {init}\nfor __idx in range({reduce_array_desc.shape[0]}):\n    __result = {op_str}",
                inputs={"__values"},
                outputs={"__result"},
            )
            self.context.state.add_edge(
                args[0].value,
                None,
                reduce_tasklet,
                "__values",
                create_memlet_full(neighbors_expr.value.data, reduce_array_desc),
            )
            self.context.state.add_edge(
                reduce_tasklet,
                "__result",
                result_access,
                None,
                dace.Memlet.simple(result_name, "0"),
            )
        else:
            assert isinstance(node.fun, itir.FunCall)
            assert isinstance(node.fun.args[0], itir.Lambda)
            fun_node = node.fun.args[0]

            args = []
            for node_arg in node.args:
                if (
                    isinstance(node_arg, itir.FunCall)
                    and isinstance(node_arg.fun, itir.SymRef)
                    and node_arg.fun.id == "neighbors"
                ):
                    expr = self.visit(node_arg)
                    args.append(*expr)
                else:
                    args.append(None)

            # first visit only arguments for neighbor selection, all other arguments are none
            neighbor_args = [arg for arg in args if arg]

            # check that all neighbors expression have the same range
            assert (
                len(
                    set([self.context.body.arrays[expr.value.data].shape for expr in neighbor_args])
                )
                == 1
            )

            nreduce = self.context.body.arrays[neighbor_args[0].value.data].shape[0]
            nreduce_domain = {"__idx": f"0:{nreduce}"}

            result_dtype = neighbor_args[0].dtype
            self.context.body.add_scalar(result_name, result_dtype, transient=True)

            assert isinstance(fun_node.expr, itir.FunCall)
            op_name = fun_node.expr.fun
            assert isinstance(op_name, itir.SymRef)

            # initialize the reduction result based on type of operation
            init_value = get_reduce_identity_value(op_name.id, result_dtype)
            init_state = self.context.body.add_state_before(self.context.state, "init", True)
            init_tasklet = init_state.add_tasklet(
                "init_reduce", {}, {"__out"}, f"__out = {init_value}"
            )
            init_state.add_edge(
                init_tasklet,
                "__out",
                init_state.add_access(result_name),
                None,
                dace.Memlet.simple(result_name, "0"),
            )

            # set reduction state to enable dereference of neighbors in input fields and to set WCR on reduce tasklet
            self.context.reduce_limit = nreduce
            self.context.reduce_wcr = "lambda x, y: " + _MATH_BUILTINS_MAPPING[str(op_name)].format(
                "x", "y"
            )

            # visit child nodes for input arguments
            for i, node_arg in enumerate(node.args):
                if not args[i]:
                    args[i] = self.visit(node_arg)[0]

            lambda_node = itir.Lambda(expr=fun_node.expr.args[1], params=fun_node.params[1:])
            lambda_context, inner_inputs, inner_outputs = self.visit(lambda_node, args=args)

            # clear context
            self.context.reduce_limit = 0
            self.context.reduce_wcr = None

            # the connectivity arrays (neighbor tables) are not needed inside the reduce lambda SDFG
            neighbor_tables = filter_neighbor_tables(self.offset_provider)
            for conn, _ in neighbor_tables:
                var = connectivity_identifier(conn)
                lambda_context.body.remove_data(var)
            # cleanup symbols previously used for shape and stride of connectivity arrays
            p = RemoveUnusedSymbols()
            p.apply_pass(lambda_context.body, {})

            input_memlets = [
                dace.Memlet.simple(expr.value.data, "__idx") for arg, expr in zip(node.args, args)
            ]
            output_memlet = dace.Memlet.simple(result_name, "0")

            input_mapping = {param: arg for (param, _), arg in zip(inner_inputs, input_memlets)}
            output_mapping = {inner_outputs[0].value.data: output_memlet}
            symbol_mapping = map_nested_sdfg_symbols(
                self.context.body, lambda_context.body, input_mapping
            )

            nsdfg_node, map_entry, _ = add_mapped_nested_sdfg(
                self.context.state,
                sdfg=lambda_context.body,
                map_ranges=nreduce_domain,
                inputs=input_mapping,
                outputs=output_mapping,
                symbol_mapping=symbol_mapping,
                input_nodes={arg.value.data: arg.value for arg in args},
                output_nodes={result_name: result_access},
            )

            # we apply map fusion only to the nested-SDFG which is generated for the reduction operator
            # the purpose is to keep the ITIR-visitor program simple and to clean up the generated SDFG
            self.context.body.apply_transformations_repeated([MapFusion], validate=False)

        return [ValueExpr(result_access, result_dtype)]

    def _visit_numeric_builtin(self, node: itir.FunCall) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        fmt = _MATH_BUILTINS_MAPPING[str(node.fun.id)]
        args: list[SymbolExpr | ValueExpr] = list(
            itertools.chain(*[self.visit(arg) for arg in node.args])
        )
        expr_args = [
            (arg, f"{arg.value.data}_v") for arg in args if not isinstance(arg, SymbolExpr)
        ]
        internals = [
            arg.value if isinstance(arg, SymbolExpr) else f"{arg.value.data}_v" for arg in args
        ]
        expr = fmt.format(*internals)
        node_type = self.node_types[id(node)]
        assert isinstance(node_type, itir_typing.Val)
        type_ = itir_type_as_dace_type(node_type.dtype)
        return self.add_expr_tasklet(expr_args, expr, type_, "numeric")

    def _visit_general_builtin(self, node: itir.FunCall) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        expr_func = _GENERAL_BUILTIN_MAPPING[str(node.fun.id)]
        return expr_func(self, node, node.args)

    def add_expr_tasklet(
        self, args: list[tuple[ValueExpr, str]], expr: str, result_type: Any, name: str
    ) -> list[ValueExpr]:
        result_name = unique_var_name()
        self.context.body.add_scalar(result_name, result_type, transient=True)
        result_access = self.context.state.add_access(result_name)

        expr_tasklet = self.context.state.add_tasklet(
            name=name,
            inputs={internal for _, internal in args},
            outputs={"__result"},
            code=f"__result = {expr}",
        )

        for arg, internal in args:
            edges = self.context.state.in_edges(expr_tasklet)
            used = False
            for edge in edges:
                if edge.dst_conn == internal:
                    used = True
                    break
            if used:
                continue
            elif not isinstance(arg, SymbolExpr):
                memlet = create_memlet_full(
                    arg.value.data, self.context.body.arrays[arg.value.data]
                )
                self.context.state.add_edge(arg.value, None, expr_tasklet, internal, memlet)

        memlet = dace.Memlet.simple(result_access.data, "0")
        self.context.state.add_edge(expr_tasklet, "__result", result_access, None, memlet)

        return [ValueExpr(result_access, result_type)]


def is_scan(node: itir.Node) -> bool:
    return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")


def closure_to_tasklet_sdfg(
    node: itir.StencilClosure,
    offset_provider: dict[str, Any],
    domain: dict[str, str],
    inputs: Sequence[tuple[str, ts.TypeSpec]],
    connectivities: Sequence[tuple[dace.ndarray, str]],
    node_types: dict[int, next_typing.Type],
) -> tuple[Context, Sequence[ValueExpr]]:
    body = dace.SDFG("tasklet_toplevel")
    state = body.add_state("tasklet_toplevel_entry", True)
    symbol_map: dict[str, TaskletExpr] = {}

    idx_accesses = {}
    for dim, idx in domain.items():
        name = f"{idx}_value"
        body.add_scalar(name, dtype=_INDEX_DTYPE, transient=True)
        tasklet = state.add_tasklet(f"get_{dim}", set(), {"value"}, f"value = {idx}")
        access = state.add_access(name)
        idx_accesses[dim] = access
        state.add_edge(tasklet, "value", access, None, dace.Memlet.simple(name, "0"))
    for name, ty in inputs:
        if isinstance(ty, ts.FieldType):
            ndim = len(ty.dims)
            shape, strides = new_array_symbols(name, ndim)
            dims = [dim.value for dim in ty.dims]
            dtype = as_dace_type(ty.dtype)
            body.add_array(name, shape=shape, strides=strides, dtype=dtype)
            field = state.add_access(name)
            indices = {dim: idx_accesses[dim] for dim in domain.keys()}
            symbol_map[name] = IteratorExpr(field, indices, dtype, dims)
        else:
            assert isinstance(ty, ts.ScalarType)
            dtype = as_dace_type(ty)
            body.add_scalar(name, dtype=dtype)
            symbol_map[name] = ValueExpr(state.add_access(name), dtype)
    for arr, name in connectivities:
        shape, strides = new_array_symbols(name, ndim=2)
        body.add_array(name, shape=shape, strides=strides, dtype=arr.dtype)

    context = Context(body, state, symbol_map)
    translator = PythonTaskletCodegen(offset_provider, context, node_types)

    args = [itir.SymRef(id=name) for name, _ in inputs]
    if is_scan(node.stencil):
        stencil = cast(FunCall, node.stencil)
        assert isinstance(stencil.args[0], Lambda)
        lambda_node = itir.Lambda(expr=stencil.args[0].expr, params=stencil.args[0].params)
        fun_node = itir.FunCall(fun=lambda_node, args=args)
    else:
        fun_node = itir.FunCall(fun=node.stencil, args=args)

    results = translator.visit(fun_node)
    for r in results:
        context.body.arrays[r.value.data].transient = False

    return context, results
