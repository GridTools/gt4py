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
from typing import Any, Callable, Optional, cast

import dace
import numpy as np
from dace.transformation.dataflow import MapFusion
from dace.transformation.passes.prune_symbols import RemoveUnusedSymbols

import gt4py.eve.codegen
from gt4py.next import Dimension, type_inference as next_typing
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
    map_nested_sdfg_symbols,
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


@dataclasses.dataclass
class SymbolExpr:
    value: str | dace.symbolic.sympy.Basic
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


@dataclasses.dataclass
class Context:
    body: dace.SDFG
    state: dace.SDFGState
    symbol_map: dict[str, IteratorExpr | ValueExpr | SymbolExpr]
    # if we encounter a reduction node, the reduction state needs to be pushed to child nodes
    reduce_limit: int
    reduce_wcr: Optional[str]

    def __init__(
        self,
        body: dace.SDFG,
        state: dace.SDFGState,
        symbol_map: dict[str, IteratorExpr | ValueExpr | SymbolExpr],
    ):
        self.body = body
        self.state = state
        self.symbol_map = symbol_map
        self.reduce_limit = 0
        self.reduce_wcr = None


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
    table_array = sdfg.arrays[table_name]

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
    sdfg.add_scalar(idx_name, dace.int64, transient=True)
    state.add_memlet_path(
        state.add_access(table_name),
        me,
        shift_tasklet,
        memlet=dace.Memlet(data=table_name, subset=",".join(f"0:{s}" for s in table_array.shape)),
        dst_conn="__table",
    )
    state.add_memlet_path(
        iterator.indices[shifted_dim],
        me,
        shift_tasklet,
        memlet=dace.Memlet(data=iterator.indices[shifted_dim].data, subset="0"),
        dst_conn="__idx",
    )
    state.add_edge(
        shift_tasklet,
        "__result",
        data_access_tasklet,
        "__idx",
        dace.Memlet(data=idx_name, subset="0"),
    )
    # select full shape only in the neighbor-axis dimension
    field_subset = [
        f"0:{sdfg.arrays[iterator.field.data].shape[idx]}"
        if dim == table.neighbor_axis.value
        else f"i_{dim}"
        for idx, dim in enumerate(iterator.dimensions)
    ]
    state.add_memlet_path(
        iterator.field,
        me,
        data_access_tasklet,
        memlet=dace.Memlet(
            data=iterator.field.data,
            subset=",".join(field_subset),
        ),
        dst_conn="__field",
    )
    state.add_memlet_path(
        data_access_tasklet,
        mx,
        result_access,
        memlet=dace.Memlet(data=result_name, subset=index_name),
        src_conn="__result",
    )

    return [ValueExpr(result_access, iterator.dtype)]


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


def builtin_undefined(*args: Any) -> Any:
    raise NotImplementedError()


_GENERAL_BUILTIN_MAPPING: dict[
    str, Callable[["PythonTaskletCodegen", itir.Expr, list[itir.Expr]], list[ValueExpr]]
] = {
    "make_tuple": builtin_make_tuple,
    "tuple_get": builtin_tuple_get,
    "if_": builtin_if,
    "cast_": builtin_cast,
    "neighbors": builtin_neighbors,
}


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
        self, node: itir.Lambda, args: Sequence[ValueExpr | SymbolExpr]
    ) -> tuple[
        Context,
        list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]],
        list[ValueExpr],
    ]:
        func_name = f"lambda_{abs(hash(node)):x}"
        neighbor_tables = filter_neighbor_tables(self.offset_provider)
        param_names = [str(p.id) for p in node.params]
        conn_names = [connectivity_identifier(offset) for offset, _ in neighbor_tables]

        assert len(param_names) == len(args)
        symbols = {
            **{param: arg for param, arg in zip(param_names, args)},
        }

        # Create the SDFG for the function's body
        prev_context = self.context
        context_sdfg = dace.SDFG(func_name)
        context_state = context_sdfg.add_state(f"{func_name}_entry", True)
        symbol_map: dict[str, ValueExpr | IteratorExpr | SymbolExpr] = {}
        value: ValueExpr | IteratorExpr
        for param, arg in symbols.items():
            if isinstance(arg, ValueExpr):
                value = ValueExpr(context_state.add_access(param), arg.dtype)
            else:
                assert isinstance(arg, IteratorExpr)
                field = context_state.add_access(param)
                indices = {
                    dim: context_state.add_access(f"__{param}_i_{dim}")
                    for dim in arg.indices.keys()
                }
                value = IteratorExpr(field, indices, arg.dtype, arg.dimensions)
            symbol_map[param] = value
        context = Context(context_sdfg, context_state, symbol_map)
        context.reduce_limit = prev_context.reduce_limit
        context.reduce_wcr = prev_context.reduce_wcr
        self.context = context

        # Add input parameters as arrays
        inputs: list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]] = []
        for name, arg in symbols.items():
            if isinstance(arg, ValueExpr):
                dtype = arg.dtype
                context.body.add_scalar(name, dtype=dtype)
                inputs.append((name, arg))
            else:
                assert isinstance(arg, IteratorExpr)
                ndims = len(arg.dimensions)
                shape = tuple(
                    dace.symbol(unique_var_name() + "__shp", dace.int64) for _ in range(ndims)
                )
                strides = tuple(
                    dace.symbol(unique_var_name() + "__strd", dace.int64) for _ in range(ndims)
                )
                dtype = arg.dtype
                context.body.add_array(name, shape=shape, strides=strides, dtype=dtype)
                index_names = {dim: f"__{name}_i_{dim}" for dim in arg.indices.keys()}
                for _, index_name in index_names.items():
                    context.body.add_scalar(index_name, dtype=dace.int64)
                inputs.append(((name, index_names), arg))

        # Add connectivities as arrays
        for name in conn_names:
            shape = (
                dace.symbol(unique_var_name() + "__shp", dace.int64),
                dace.symbol(unique_var_name() + "__shp", dace.int64),
            )
            strides = (
                dace.symbol(unique_var_name() + "__strd", dace.int64),
                dace.symbol(unique_var_name() + "__strd", dace.int64),
            )
            dtype = prev_context.body.arrays[name].dtype
            context.body.add_array(name, shape=shape, strides=strides, dtype=dtype)

        # Translate the function's body
        result: ValueExpr | SymbolExpr = self.visit(node.expr)[0]
        # Forwarding result through a tasklet needed because empty SDFG states don't properly forward connectors
        if isinstance(result, ValueExpr):
            result_name = unique_var_name()
            self.context.body.add_scalar(result_name, result.dtype, transient=True)
            result_access = self.context.state.add_access(result_name)
            self.context.state.add_edge(
                result.value,
                None,
                result_access,
                None,
                # in case of reduction lambda, the output edge from lambda tasklet performs write-conflict resolution
                dace.Memlet(f"{result_access.data}[0]", wcr=context.reduce_wcr),
            )
            result = ValueExpr(value=result_access, dtype=result.dtype)
        else:
            result = self.add_expr_tasklet([], result.value, result.dtype, "forward")[0]
        self.context.body.arrays[result.value.data].transient = False
        self.context = prev_context

        for node in context.state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if context.state.out_degree(node) == 0 and context.state.in_degree(node) == 0:
                    context.state.remove_node(node)

        return context, inputs, [result]

    def visit_SymRef(self, node: itir.SymRef) -> list[ValueExpr | SymbolExpr] | IteratorExpr:
        if node.id not in self.context.symbol_map:
            acc = self.context.state.add_access(node.id)
            node_type = self.node_types[id(node)]
            assert isinstance(node_type, Val)
            self.context.symbol_map[node.id] = ValueExpr(
                value=acc, dtype=itir_type_as_dace_type(node_type.dtype)
            )
        value = self.context.symbol_map[node.id]
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
                offset = node.fun.args[0]
                assert isinstance(offset, itir.OffsetLiteral)
                offset_name = offset.value
                assert isinstance(offset_name, str)
                if offset_name not in self.offset_provider:
                    raise ValueError(f"offset provider for `{offset_name}` is missing")
                offset_provider = self.offset_provider[offset_name]
                if isinstance(offset_provider, Dimension):
                    return self._visit_direct_addressing(node)
                else:
                    return self._visit_indirect_addressing(node)
            elif node.fun.fun.id == "reduce":
                return self._visit_reduce(node)

        if isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in _MATH_BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            elif str(node.fun.id) in _GENERAL_BUILTIN_MAPPING:
                return self._visit_general_builtin(node)
            else:
                raise NotImplementedError()
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
            array_index = [
                f"{iterator.indices[dim].data}_v" if dim in iterator.indices else index_name
                for dim in sorted(iterator.dimensions)
            ]
            args = [ValueExpr(iterator.field, iterator.dtype)] + [
                ValueExpr(iterator.indices[dim], iterator.dtype) for dim in iterator.indices
            ]
            internals = [f"{arg.value.data}_v" for arg in args]

            deref_tasklet = self.context.state.add_tasklet(
                name="deref",
                inputs=set(internals),
                outputs={"__result"},
                code=f"__result = {args[0].value.data}_v[{', '.join(array_index)}]",
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
                memlet=dace.Memlet(data=result_name, subset=index_name),
                src_conn="__result",
            )

            return [ValueExpr(value=result_access, dtype=iterator.dtype)]

        else:
            sorted_index = sorted(iterator.indices.items(), key=lambda x: x[0])
            flat_index = [
                ValueExpr(x[1], iterator.dtype) for x in sorted_index if x[0] in iterator.dimensions
            ]

            args = [ValueExpr(iterator.field, int), *flat_index]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{', '.join(internals[1:])}]"
            return self.add_expr_tasklet(list(zip(args, internals)), expr, iterator.dtype, "deref")

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

    def _visit_direct_addressing(self, node: itir.FunCall) -> IteratorExpr:
        assert isinstance(node.fun, itir.FunCall)
        shift = node.fun
        assert isinstance(shift, itir.FunCall)

        tail, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0])

        assert isinstance(tail[0], itir.OffsetLiteral)
        offset = tail[0].value
        assert isinstance(offset, str)
        shifted_dim = self.offset_provider[offset].value

        assert isinstance(tail[1], itir.OffsetLiteral)
        shift_amount = tail[1].value
        assert isinstance(shift_amount, int)

        args = [ValueExpr(iterator.indices[shifted_dim], dace.int64)]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[0]} + {shift_amount}"
        shifted_value = self.add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "dir_addr"
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        shifted_index[shifted_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index, iterator.dtype, iterator.dimensions)

    def _visit_indirect_addressing(self, node: itir.FunCall) -> IteratorExpr:
        shift = node.fun
        assert isinstance(shift, itir.FunCall)
        tail, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0])

        assert isinstance(tail[0], itir.OffsetLiteral)
        offset = tail[0].value
        assert isinstance(offset, str)

        assert isinstance(tail[1], itir.OffsetLiteral)
        element = tail[1].value
        assert isinstance(element, int)

        table: NeighborTableOffsetProvider = self.offset_provider[offset]
        shifted_dim = table.origin_axis.value
        target_dim = table.neighbor_axis.value

        conn = self.context.state.add_access(connectivity_identifier(offset))

        args = [
            ValueExpr(conn, table.table.dtype),
            ValueExpr(iterator.indices[shifted_dim], dace.int64),
        ]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[0]}[{internals[1]}, {element}]"
        shifted_value = self.add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "ind_addr"
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        del shifted_index[shifted_dim]
        shifted_index[target_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index, iterator.dtype, iterator.dimensions)

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

            nreduce = self.context.body.arrays[neighbors_expr.value.data].shape[0]

            self.context.body.add_scalar(result_name, result_dtype, transient=True)
            op_str = _MATH_BUILTINS_MAPPING[str(op_name)].format("__result", "__values[__idx]")
            reduce_tasklet = self.context.state.add_tasklet(
                "reduce",
                code=f"__result = {init}\nfor __idx in range({nreduce}):\n    __result = {op_str}",
                inputs={"__values"},
                outputs={"__result"},
            )
            self.context.state.add_edge(
                args[0].value,
                None,
                reduce_tasklet,
                "__values",
                dace.Memlet(data=neighbors_expr.value.data, subset=f"0:{nreduce}"),
            )
            self.context.state.add_edge(
                reduce_tasklet,
                "__result",
                result_access,
                None,
                dace.Memlet(data=result_name, subset="0"),
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
            init_state = self.context.body.add_state_before(self.context.state, "init")
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
                create_memlet_at(expr.value.data, ("__idx",)) for arg, expr in zip(node.args, args)
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

        memlet = create_memlet_at(result_access.data, ("0",))
        self.context.state.add_edge(expr_tasklet, "__result", result_access, None, memlet)

        return [ValueExpr(result_access, result_type)]


def is_scan(node: itir.Node) -> bool:
    return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")


def _visit_scan_closure_callable(
    node: itir.StencilClosure,
    tlet_codegen: PythonTaskletCodegen,
) -> tuple[Context, Sequence[tuple[str, ValueExpr]], Sequence[ValueExpr]]:
    stencil = cast(FunCall, node.stencil)
    assert isinstance(stencil.args[0], Lambda)
    fun_node = itir.Lambda(expr=stencil.args[0].expr, params=stencil.args[0].params)

    args = list(itertools.chain(tlet_codegen.visit(node.output), *tlet_codegen.visit(node.inputs)))
    return tlet_codegen.visit(fun_node, args=args)


def _visit_closure_callable(
    node: itir.StencilClosure,
    tlet_codegen: PythonTaskletCodegen,
    input_names: Sequence[str],
) -> Sequence[ValueExpr]:
    args = [itir.SymRef(id=name) for name in input_names]
    fun_node = itir.FunCall(fun=node.stencil, args=args)

    return tlet_codegen.visit(fun_node)


def closure_to_tasklet_sdfg(
    node: itir.StencilClosure,
    offset_provider: dict[str, Any],
    domain: dict[str, str],
    inputs: Sequence[tuple[str, ts.TypeSpec]],
    connectivities: Sequence[tuple[dace.ndarray, str]],
    node_types: dict[int, next_typing.Type],
) -> tuple[Context, Sequence[tuple[str, ValueExpr]], Sequence[ValueExpr]]:
    body = dace.SDFG("tasklet_toplevel")
    state = body.add_state("tasklet_toplevel_entry")
    symbol_map: dict[str, ValueExpr | IteratorExpr | SymbolExpr] = {}

    idx_accesses = {}
    for dim, idx in domain.items():
        name = f"{idx}_value"
        body.add_scalar(name, dtype=dace.int64, transient=True)
        tasklet = state.add_tasklet(f"get_{dim}", set(), {"value"}, f"value = {idx}")
        access = state.add_access(name)
        idx_accesses[dim] = access
        state.add_edge(tasklet, "value", access, None, dace.Memlet(data=name, subset="0"))
    for name, ty in inputs:
        if isinstance(ty, ts.FieldType):
            ndim = len(ty.dims)
            shape = [
                dace.symbol(f"{unique_var_name()}_shp{i}", dtype=dace.int64) for i in range(ndim)
            ]
            stride = [
                dace.symbol(f"{unique_var_name()}_strd{i}", dtype=dace.int64) for i in range(ndim)
            ]
            dims = [dim.value for dim in ty.dims]
            dtype = as_dace_type(ty.dtype)
            body.add_array(name, shape=shape, strides=stride, dtype=dtype)
            field = state.add_access(name)
            indices = {dim: idx_accesses[dim] for dim in domain.keys()}
            symbol_map[name] = IteratorExpr(field, indices, dtype, dims)
        else:
            assert isinstance(ty, ts.ScalarType)
            dtype = as_dace_type(ty)
            body.add_scalar(name, dtype=dtype)
            symbol_map[name] = ValueExpr(state.add_access(name), dtype)
    for arr, name in connectivities:
        shape = [dace.symbol(f"{unique_var_name()}_shp{i}", dtype=dace.int64) for i in range(2)]
        stride = [dace.symbol(f"{unique_var_name()}_strd{i}", dtype=dace.int64) for i in range(2)]
        body.add_array(name, shape=shape, strides=stride, dtype=arr.dtype)

    context = Context(body, state, symbol_map)
    translator = PythonTaskletCodegen(offset_provider, context, node_types)

    if is_scan(node.stencil):
        context, inner_inputs, inner_outputs = _visit_scan_closure_callable(node, translator)
    else:
        inner_inputs = []
        inner_outputs = _visit_closure_callable(
            node,
            translator,
            [name for name, _ in inputs],
        )
    for output in inner_outputs:
        context.body.arrays[output.value.data].transient = False

    return context, inner_inputs, inner_outputs
