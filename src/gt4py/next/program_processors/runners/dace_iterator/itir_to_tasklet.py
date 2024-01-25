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
from dace.transformation.dataflow import MapFusion

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
    dace_debuginfo,
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
    reduce_identity: Optional[SymbolExpr]

    def __init__(
        self,
        body: dace.SDFG,
        state: dace.SDFGState,
        symbol_map: dict[str, TaskletExpr],
        reduce_identity: Optional[SymbolExpr] = None,
    ):
        self.body = body
        self.state = state
        self.symbol_map = symbol_map
        self.reduce_identity = reduce_identity


def builtin_neighbors(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_debuginfo(node, transformer.context.body.debuginfo)
    offset_literal, data = node_args
    assert isinstance(offset_literal, itir.OffsetLiteral)
    offset_dim = offset_literal.value
    assert isinstance(offset_dim, str)
    iterator = transformer.visit(data)
    assert isinstance(iterator, IteratorExpr)
    field_desc = iterator.field.desc(transformer.context.body)

    field_index = "__field_idx"
    offset_provider = transformer.offset_provider[offset_dim]
    if isinstance(offset_provider, NeighborTableOffsetProvider):
        neighbor_check = f"{field_index} >= 0"
    elif isinstance(offset_provider, StridedNeighborOffsetProvider):
        neighbor_check = f"{field_index} < {field_desc.shape[offset_provider.neighbor_axis.value]}"
    else:
        assert isinstance(offset_provider, Dimension)
        raise NotImplementedError(
            "Neighbor reductions for cartesian grids not implemented in DaCe backend."
        )

    assert transformer.context.reduce_identity is not None

    sdfg: dace.SDFG = transformer.context.body
    state: dace.SDFGState = transformer.context.state

    shifted_dim = offset_provider.origin_axis.value

    result_name = unique_var_name()
    sdfg.add_array(
        result_name, dtype=iterator.dtype, shape=(offset_provider.max_neighbors,), transient=True
    )
    result_access = state.add_access(result_name, debuginfo=di)

    # generate unique map index name to avoid conflict with other maps inside same state
    neighbor_index = unique_name("neighbor_idx")
    me, mx = state.add_map(
        f"{offset_dim}_neighbors_map",
        ndrange={neighbor_index: f"0:{offset_provider.max_neighbors}"},
        debuginfo=di,
    )
    table_name = connectivity_identifier(offset_dim)
    table_subset = (f"0:{sdfg.arrays[table_name].shape[0]}", neighbor_index)

    shift_tasklet = state.add_tasklet(
        "shift",
        code="__result = __table[__idx]",
        inputs={"__table", "__idx"},
        outputs={"__result"},
        debuginfo=di,
    )
    data_access_tasklet = state.add_tasklet(
        "data_access",
        code=f"__result = __field[{field_index}] if {neighbor_check} else {transformer.context.reduce_identity.value}",
        inputs={"__field", field_index},
        outputs={"__result"},
        debuginfo=di,
    )
    idx_name = unique_var_name()
    sdfg.add_scalar(idx_name, _INDEX_DTYPE, transient=True)
    state.add_memlet_path(
        state.add_access(table_name, debuginfo=di),
        me,
        shift_tasklet,
        memlet=create_memlet_at(table_name, table_subset),
        dst_conn="__table",
    )
    state.add_memlet_path(
        iterator.indices[shifted_dim],
        me,
        shift_tasklet,
        memlet=dace.Memlet.simple(iterator.indices[shifted_dim].data, "0", debuginfo=di),
        dst_conn="__idx",
    )
    state.add_edge(shift_tasklet, "__result", data_access_tasklet, field_index, dace.Memlet())
    # select full shape only in the neighbor-axis dimension
    field_subset = tuple(
        f"0:{shape}" if dim == offset_provider.neighbor_axis.value else f"i_{dim}"
        for dim, shape in zip(sorted(iterator.dimensions), field_desc.shape)
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
        memlet=dace.Memlet.simple(result_name, neighbor_index, debuginfo=di),
        src_conn="__result",
    )

    return [ValueExpr(result_access, iterator.dtype)]


def builtin_can_deref(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_debuginfo(node, transformer.context.body.debuginfo)
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
        result_node = transformer.context.state.add_access(result_name, debuginfo=di)
        transformer.context.state.add_edge(
            transformer.context.state.add_tasklet(
                "can_always_deref", {}, {"_out"}, "_out = True", debuginfo=di
            ),
            "_out",
            result_node,
            None,
            dace.Memlet.simple(result_name, "0", debuginfo=di),
        )
        return [ValueExpr(result_node, dace.dtypes.bool)]

    # create tasklet to check that field indices are non-negative (-1 is invalid)
    args = [ValueExpr(access_node, _INDEX_DTYPE) for access_node in iterator.indices.values()]
    internals = [f"{arg.value.data}_v" for arg in args]
    expr_code = " and ".join([f"{v} >= 0" for v in internals])

    # TODO(edopao): select-memlet could maybe allow to efficiently translate can_deref to predicative execution
    return transformer.add_expr_tasklet(
        list(zip(args, internals)),
        expr_code,
        dace.dtypes.bool,
        "can_deref",
        dace_debuginfo=di,
    )


def builtin_if(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_debuginfo(node, transformer.context.body.debuginfo)
    args = transformer.visit(node_args)
    assert len(args) == 3
    if_node = args[0][0] if isinstance(args[0], list) else args[0]

    # the argument could be a list of elements on each branch representing the result of `make_tuple`
    # however, the normal case is to find one value expression
    assert len(args[1]) == len(args[2])
    if_expr_args = [
        (a[0] if isinstance(a, list) else a, b[0] if isinstance(b, list) else b)
        for a, b in zip(args[1], args[2])
    ]

    # in case of tuple arguments, generate one if-tasklet for each element of the output tuple
    if_expr_values = []
    for a, b in if_expr_args:
        assert a.dtype == b.dtype
        expr_args = [
            (arg, f"{arg.value.data}_v")
            for arg in (if_node, a, b)
            if not isinstance(arg, SymbolExpr)
        ]
        internals = [
            arg.value if isinstance(arg, SymbolExpr) else f"{arg.value.data}_v"
            for arg in (if_node, a, b)
        ]
        expr = "({1} if {0} else {2})".format(*internals)
        if_expr = transformer.add_expr_tasklet(expr_args, expr, a.dtype, "if", dace_debuginfo=di)
        if_expr_values.append(if_expr[0])

    return if_expr_values


def builtin_list_get(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_debuginfo(node, transformer.context.body.debuginfo)
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
    return transformer.add_expr_tasklet(
        expr_args, expr, args[1].dtype, "list_get", dace_debuginfo=di
    )


def builtin_cast(
    transformer: "PythonTaskletCodegen", node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_debuginfo(node, transformer.context.body.debuginfo)
    args = transformer.visit(node_args[0])
    internals = [f"{arg.value.data}_v" for arg in args]
    target_type = node_args[1]
    assert isinstance(target_type, itir.SymRef)
    expr = _MATH_BUILTINS_MAPPING[target_type.id].format(*internals)
    node_type = transformer.node_types[id(node)]
    assert isinstance(node_type, itir_typing.Val)
    type_ = itir_type_as_dace_type(node_type.dtype)
    return transformer.add_expr_tasklet(
        list(zip(args, internals)),
        expr,
        type_,
        "cast",
        dace_debuginfo=di,
    )


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
        return [elements[int(index.value)]]
    raise ValueError("Tuple can only be subscripted with compile-time constants.")


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
            self._symbol_map[param] = ValueExpr(
                self._state.add_access(param, debuginfo=self._sdfg.debuginfo), arg.dtype
            )
        elif isinstance(arg, IteratorExpr):
            # create storage in lambda sdfg
            ndims = len(arg.dimensions)
            shape, strides = new_array_symbols(param, ndims)
            self._sdfg.add_array(param, shape=shape, strides=strides, dtype=arg.dtype)
            index_names = {dim: f"__{param}_i_{dim}" for dim in arg.indices.keys()}
            for _, index_name in index_names.items():
                self._sdfg.add_scalar(index_name, dtype=_INDEX_DTYPE)
            # update table of lambda symbol
            field = self._state.add_access(param, debuginfo=self._sdfg.debuginfo)
            indices = {
                dim: self._state.add_access(index_arg, debuginfo=self._sdfg.debuginfo)
                for dim, index_arg in index_names.items()
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
            access_node = self._state.add_access(param, debuginfo=self._sdfg.debuginfo)
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
        self, node: itir.Lambda, args: Sequence[TaskletExpr], use_neighbor_tables: bool = True
    ) -> tuple[
        Context,
        list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]],
        list[ValueExpr],
    ]:
        func_name = f"lambda_{abs(hash(node)):x}"
        neighbor_tables = (
            filter_neighbor_tables(self.offset_provider) if use_neighbor_tables else []
        )
        connectivity_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

        # Create the SDFG for the lambda's body
        lambda_sdfg = dace.SDFG(func_name)
        lambda_sdfg.debuginfo = dace_debuginfo(node)
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
            reduce_identity=self.context.reduce_identity,
        )
        lambda_taskgen = PythonTaskletCodegen(self.offset_provider, lambda_context, self.node_types)

        results: list[ValueExpr] = []
        # We are flattening the returned list of value expressions because the multiple outputs of a lambda
        # should be a list of nodes without tuple structure. Ideally, an ITIR transformation could do this.
        node.expr.location = node.location
        for expr in flatten_list(lambda_taskgen.visit(node.expr)):
            if isinstance(expr, ValueExpr):
                result_name = unique_var_name()
                lambda_sdfg.add_scalar(result_name, expr.dtype, transient=True)
                result_access = lambda_state.add_access(
                    result_name, debuginfo=lambda_sdfg.debuginfo
                )
                lambda_state.add_nedge(
                    expr.value,
                    result_access,
                    dace.Memlet.simple(result_access.data, "0"),
                )
                result = ValueExpr(value=result_access, dtype=expr.dtype)
            else:
                # Forwarding result through a tasklet needed because empty SDFG states don't properly forward connectors
                result = lambda_taskgen.add_expr_tasklet(
                    [], expr.value, expr.dtype, "forward", dace_debuginfo=lambda_sdfg.debuginfo
                )[0]
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
        node.fun.location = node.location
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
                raise NotImplementedError(f"'{builtin_name}' not implemented.")
        return self._visit_call(node)

    def _visit_call(self, node: itir.FunCall):
        args = self.visit(node.args)
        args = [arg if isinstance(arg, Sequence) else [arg] for arg in args]
        args = list(itertools.chain(*args))
        node.fun.location = node.location
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
            debuginfo=dace_debuginfo(node, func_context.body.debuginfo),
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
            access = self.context.state.add_access(var, debuginfo=nsdfg_node.debuginfo)
            self.context.state.add_edge(access, None, nsdfg_node, var, memlet)

        result_exprs = []
        for result in results:
            name = unique_var_name()
            self.context.body.add_scalar(name, result.dtype, transient=True)
            result_access = self.context.state.add_access(name, debuginfo=nsdfg_node.debuginfo)
            result_exprs.append(ValueExpr(result_access, result.dtype))
            memlet = create_memlet_full(name, self.context.body.arrays[name])
            self.context.state.add_edge(nsdfg_node, result.value.data, result_access, None, memlet)

        return result_exprs

    def _visit_deref(self, node: itir.FunCall) -> list[ValueExpr]:
        di = dace_debuginfo(node, self.context.body.debuginfo)
        iterator = self.visit(node.args[0])
        if not isinstance(iterator, IteratorExpr):
            # already a list of ValueExpr
            return iterator

        args: list[ValueExpr]
        sorted_dims = sorted(iterator.dimensions)
        if all([dim in iterator.indices for dim in iterator.dimensions]):
            # The deref iterator has index values on all dimensions: the result will be a scalar
            args = [ValueExpr(iterator.field, iterator.dtype)] + [
                ValueExpr(iterator.indices[dim], _INDEX_DTYPE) for dim in sorted_dims
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{', '.join(internals[1:])}]"
            return self.add_expr_tasklet(
                list(zip(args, internals)),
                expr,
                iterator.dtype,
                "deref",
                dace_debuginfo=di,
            )

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
            result_node = self.context.state.add_access(result_name, debuginfo=di)

            deref_connectors = ["_inp"] + [
                f"_i_{dim}" for dim in sorted_dims if dim in iterator.indices
            ]
            deref_nodes = [iterator.field] + [
                iterator.indices[dim] for dim in sorted_dims if dim in iterator.indices
            ]
            deref_memlets = [dace.Memlet.from_array(iterator.field.data, field_array)] + [
                dace.Memlet.simple(node.data, "0") for node in deref_nodes[1:]
            ]

            # we create a mapped tasklet for array slicing
            map_ranges = {
                f"_i_{dim}": f"0:{size}"
                for dim, size in zip(sorted_dims, field_array.shape)
                if dim not in iterator.indices
            }
            src_subset = ",".join([f"_i_{dim}" for dim in sorted_dims])
            dst_subset = ",".join(
                [f"_i_{dim}" for dim in sorted_dims if dim not in iterator.indices]
            )
            self.context.state.add_mapped_tasklet(
                "deref",
                map_ranges,
                inputs={k: v for k, v in zip(deref_connectors, deref_memlets)},
                outputs={
                    "_out": dace.Memlet.from_array(result_name, result_array),
                },
                code=f"_out[{dst_subset}] = _inp[{src_subset}]",
                external_edges=True,
                input_nodes={node.data: node for node in deref_nodes},
                output_nodes={
                    result_name: result_node,
                },
                debuginfo=di,
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
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest),
            args=[iterator],
            location=iterator.location,
        )

    def _visit_shift(self, node: itir.FunCall) -> IteratorExpr | list[ValueExpr]:
        di = dace_debuginfo(node, self.context.body.debuginfo)
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
            connectivity = self.context.state.add_access(
                connectivity_identifier(offset_dim), debuginfo=di
            )

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
            list(zip(args, internals)), expr, offset_node.dtype, "shift", dace_debuginfo=di
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        del shifted_index[shifted_dim]
        shifted_index[target_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index, iterator.dtype, iterator.dimensions)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral) -> list[ValueExpr]:
        di = dace_debuginfo(node, self.context.body.debuginfo)
        offset = node.value
        assert isinstance(offset, int)
        offset_var = unique_var_name()
        self.context.body.add_scalar(offset_var, _INDEX_DTYPE, transient=True)
        offset_node = self.context.state.add_access(offset_var, debuginfo=di)
        tasklet_node = self.context.state.add_tasklet(
            "get_offset", {}, {"__out"}, f"__out = {offset}", debuginfo=di
        )
        self.context.state.add_edge(
            tasklet_node, "__out", offset_node, None, dace.Memlet.simple(offset_var, "0")
        )
        return [ValueExpr(offset_node, self.context.body.arrays[offset_var].dtype)]

    def _visit_reduce(self, node: itir.FunCall):
        di = dace_debuginfo(node, self.context.body.debuginfo)
        node_type = self.node_types[id(node)]
        assert isinstance(node_type, itir_typing.Val)
        reduce_dtype = itir_type_as_dace_type(node_type.dtype)

        if len(node.args) == 1:
            assert (
                isinstance(node.args[0], itir.FunCall)
                and isinstance(node.args[0].fun, itir.SymRef)
                and node.args[0].fun.id == "neighbors"
            )
            assert isinstance(node.fun, itir.FunCall)
            op_name = node.fun.args[0]
            assert isinstance(op_name, itir.SymRef)
            reduce_identity = node.fun.args[1]
            assert isinstance(reduce_identity, itir.Literal)

            # set reduction state
            self.context.reduce_identity = SymbolExpr(reduce_identity, reduce_dtype)

            args = self.visit(node.args)

            assert len(args) == 1 and len(args[0]) == 1
            reduce_input_node = args[0][0].value

        else:
            assert isinstance(node.fun, itir.FunCall)
            assert isinstance(node.fun.args[0], itir.Lambda)
            fun_node = node.fun.args[0]
            assert isinstance(fun_node.expr, itir.FunCall)

            op_name = fun_node.expr.fun
            assert isinstance(op_name, itir.SymRef)
            reduce_identity = get_reduce_identity_value(op_name.id, reduce_dtype)

            # set reduction state in visit context
            self.context.reduce_identity = SymbolExpr(reduce_identity, reduce_dtype)

            args = flatten_list(self.visit(node.args))

            # clear context
            self.context.reduce_identity = None

            # check that all neighbor expressions have the same shape
            nreduce_shape = args[1].value.desc(self.context.body).shape
            assert all(
                [arg.value.desc(self.context.body).shape == nreduce_shape for arg in args[2:]]
            )

            nreduce_index = tuple(f"_i{i}" for i in range(len(nreduce_shape)))
            nreduce_domain = {idx: f"0:{size}" for idx, size in zip(nreduce_index, nreduce_shape)}

            reduce_input_name = unique_var_name()
            self.context.body.add_array(
                reduce_input_name, nreduce_shape, reduce_dtype, transient=True
            )

            lambda_node = itir.Lambda(
                expr=fun_node.expr.args[1], params=fun_node.params[1:], location=node.location
            )
            lambda_context, inner_inputs, inner_outputs = self.visit(
                lambda_node, args=args, use_neighbor_tables=False
            )

            input_mapping = {
                param: create_memlet_at(arg.value.data, nreduce_index)
                for (param, _), arg in zip(inner_inputs, args)
            }
            output_mapping = {
                inner_outputs[0].value.data: create_memlet_at(reduce_input_name, nreduce_index)
            }
            symbol_mapping = map_nested_sdfg_symbols(
                self.context.body, lambda_context.body, input_mapping
            )

            reduce_input_node = self.context.state.add_access(reduce_input_name, debuginfo=di)

            nsdfg_node, map_entry, _ = add_mapped_nested_sdfg(
                self.context.state,
                sdfg=lambda_context.body,
                map_ranges=nreduce_domain,
                inputs=input_mapping,
                outputs=output_mapping,
                symbol_mapping=symbol_mapping,
                input_nodes={arg.value.data: arg.value for arg in args},
                output_nodes={reduce_input_name: reduce_input_node},
                debuginfo=di,
            )

        reduce_input_desc = reduce_input_node.desc(self.context.body)

        result_name = unique_var_name()
        # we allocate an array instead of a scalar because the reduce library node is generic and expects an array node
        self.context.body.add_array(result_name, (1,), reduce_dtype, transient=True)
        result_access = self.context.state.add_access(result_name, debuginfo=di)

        reduce_wcr = "lambda x, y: " + _MATH_BUILTINS_MAPPING[str(op_name)].format("x", "y")
        reduce_node = self.context.state.add_reduce(reduce_wcr, None, reduce_identity)
        self.context.state.add_nedge(
            reduce_input_node,
            reduce_node,
            dace.Memlet.from_array(reduce_input_node.data, reduce_input_desc),
        )
        self.context.state.add_nedge(
            reduce_node, result_access, dace.Memlet.simple(result_name, "0")
        )

        # we apply map fusion only to the nested-SDFG which is generated for the reduction operator
        # the purpose is to keep the ITIR-visitor program simple and to clean up the generated SDFG
        self.context.body.apply_transformations_repeated([MapFusion], validate=False)

        return [ValueExpr(result_access, reduce_dtype)]

    def _visit_numeric_builtin(self, node: itir.FunCall) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        fmt = _MATH_BUILTINS_MAPPING[str(node.fun.id)]
        args = flatten_list(self.visit(node.args))
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
        return self.add_expr_tasklet(
            expr_args,
            expr,
            type_,
            "numeric",
            dace_debuginfo=dace_debuginfo(node, self.context.body.debuginfo),
        )

    def _visit_general_builtin(self, node: itir.FunCall) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        expr_func = _GENERAL_BUILTIN_MAPPING[str(node.fun.id)]
        return expr_func(self, node, node.args)

    def add_expr_tasklet(
        self,
        args: list[tuple[ValueExpr, str]],
        expr: str,
        result_type: Any,
        name: str,
        dace_debuginfo: Optional[dace.dtypes.DebugInfo] = None,
    ) -> list[ValueExpr]:
        di = dace_debuginfo if dace_debuginfo else self.context.body.debuginfo
        result_name = unique_var_name()
        self.context.body.add_scalar(result_name, result_type, transient=True)
        result_access = self.context.state.add_access(result_name, debuginfo=di)

        expr_tasklet = self.context.state.add_tasklet(
            name=name,
            inputs={internal for _, internal in args},
            outputs={"__result"},
            code=f"__result = {expr}",
            debuginfo=di,
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

        memlet = dace.Memlet.simple(result_access.data, "0", debuginfo=di)
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
    body.debuginfo = dace_debuginfo(node)
    state = body.add_state("tasklet_toplevel_entry", True)
    symbol_map: dict[str, TaskletExpr] = {}

    idx_accesses = {}
    for dim, idx in domain.items():
        name = f"{idx}_value"
        body.add_scalar(name, dtype=_INDEX_DTYPE, transient=True)
        tasklet = state.add_tasklet(
            f"get_{dim}", set(), {"value"}, f"value = {idx}", debuginfo=body.debuginfo
        )
        access = state.add_access(name, debuginfo=body.debuginfo)
        idx_accesses[dim] = access
        state.add_edge(tasklet, "value", access, None, dace.Memlet.simple(name, "0"))
    for name, ty in inputs:
        if isinstance(ty, ts.FieldType):
            ndim = len(ty.dims)
            shape, strides = new_array_symbols(name, ndim)
            dims = [dim.value for dim in ty.dims]
            dtype = as_dace_type(ty.dtype)
            body.add_array(name, shape=shape, strides=strides, dtype=dtype)
            field = state.add_access(name, debuginfo=body.debuginfo)
            indices = {dim: idx_accesses[dim] for dim in domain.keys()}
            symbol_map[name] = IteratorExpr(field, indices, dtype, dims)
        else:
            assert isinstance(ty, ts.ScalarType)
            dtype = as_dace_type(ty)
            body.add_scalar(name, dtype=dtype)
            symbol_map[name] = ValueExpr(state.add_access(name, debuginfo=body.debuginfo), dtype)
    for arr, name in connectivities:
        shape, strides = new_array_symbols(name, ndim=2)
        body.add_array(name, shape=shape, strides=strides, dtype=arr.dtype)

    context = Context(body, state, symbol_map)
    translator = PythonTaskletCodegen(offset_provider, context, node_types)

    args = [itir.SymRef(id=name) for name, _ in inputs]
    if is_scan(node.stencil):
        stencil = cast(FunCall, node.stencil)
        assert isinstance(stencil.args[0], Lambda)
        lambda_node = itir.Lambda(
            expr=stencil.args[0].expr, params=stencil.args[0].params, location=node.location
        )
        fun_node = itir.FunCall(fun=lambda_node, args=args, location=node.location)
    else:
        fun_node = itir.FunCall(fun=node.stencil, args=args, location=node.location)

    results = translator.visit(fun_node)
    for r in results:
        context.body.arrays[r.value.data].transient = False

    return context, results
