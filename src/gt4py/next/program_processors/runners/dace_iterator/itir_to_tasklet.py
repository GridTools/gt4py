# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeAlias, cast

import dace
import numpy as np

import gt4py.eve.codegen
from gt4py import eve
from gt4py.next import Dimension
from gt4py.next.common import _DEFAULT_SKIP_VALUE as neighbor_skip_value, Connectivity
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir import FunCall, Lambda
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.program_processors.runners.dace_common import utility as dace_utils
from gt4py.next.type_system import type_specifications as ts

from .utility import (
    add_mapped_nested_sdfg,
    flatten_list,
    get_used_connectivities,
    map_nested_sdfg_symbols,
    new_array_symbols,
    unique_name,
    unique_var_name,
)


_TYPE_MAPPING = {
    "float": dace.float64,
    "float16": dace.float16,
    "float32": dace.float32,
    "float64": dace.float64,
    "int": dace.int32 if np.dtype(int).itemsize == 4 else dace.int64,
    "int8": dace.int8,
    "uint8": dace.uint8,
    "int16": dace.int16,
    "uint16": dace.uint16,
    "int32": dace.int32,
    "uint32": dace.uint32,
    "int64": dace.int64,
    "uint64": dace.uint64,
    "bool": dace.bool_,
}


def itir_type_as_dace_type(type_: ts.TypeSpec):
    # TODO(tehrengruber): this function just converts the scalar type of whatever it is given,
    #  let it be a field, iterator, or directly a scalar. The caller should take care of the
    #  extraction.
    dtype: ts.TypeSpec
    if isinstance(type_, ts.FieldType):
        dtype = type_.dtype
    elif isinstance(type_, it_ts.IteratorType):
        dtype = type_.element_type
    else:
        dtype = type_
    assert isinstance(dtype, ts.ScalarType)
    return _TYPE_MAPPING[dtype.kind.name.lower()]


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
    "float16": "dace.float16({})",
    "float32": "dace.float32({})",
    "float64": "dace.float64({})",
    "int": "dace.int32({})" if np.dtype(int).itemsize == 4 else "dace.int64({})",
    "int8": "dace.int8({})",
    "uint8": "dace.uint8({})",
    "int16": "dace.int16({})",
    "uint16": "dace.uint16({})",
    "int32": "dace.int32({})",
    "uint32": "dace.uint32({})",
    "int64": "dace.int64({})",
    "uint64": "dace.uint64({})",
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


def _visit_lift_in_neighbors_reduction(
    transformer: PythonTaskletCodegen,
    node: itir.FunCall,
    node_args: Sequence[IteratorExpr | list[ValueExpr]],
    offset_provider: Connectivity,
    map_entry: dace.nodes.MapEntry,
    map_exit: dace.nodes.MapExit,
    neighbor_index_node: dace.nodes.AccessNode,
    neighbor_value_node: dace.nodes.AccessNode,
) -> list[ValueExpr]:
    assert transformer.context.reduce_identity is not None
    neighbor_dim = offset_provider.neighbor_axis.value
    origin_dim = offset_provider.origin_axis.value

    lifted_args: list[IteratorExpr | ValueExpr] = []
    for arg in node_args:
        if isinstance(arg, IteratorExpr):
            if origin_dim in arg.indices:
                lifted_indices = arg.indices.copy()
                lifted_indices.pop(origin_dim)
                lifted_indices[neighbor_dim] = neighbor_index_node
                lifted_args.append(
                    IteratorExpr(arg.field, lifted_indices, arg.dtype, arg.dimensions)
                )
            else:
                lifted_args.append(arg)
        else:
            lifted_args.append(arg[0])

    lift_context, inner_inputs, inner_outputs = transformer.visit(node.args[0], args=lifted_args)
    assert len(inner_outputs) == 1
    inner_out_connector = inner_outputs[0].value.data

    input_nodes = {}
    iterator_index_nodes = {}
    lifted_index_connectors = []

    for x, y in inner_inputs:
        if isinstance(y, IteratorExpr):
            field_connector, inner_index_table = x
            input_nodes[field_connector] = y.field
            for dim, connector in inner_index_table.items():
                if dim == neighbor_dim:
                    lifted_index_connectors.append(connector)
                iterator_index_nodes[connector] = y.indices[dim]
        else:
            assert isinstance(y, ValueExpr)
            input_nodes[x] = y.value

    neighbor_tables = get_used_connectivities(node.args[0], transformer.offset_provider)
    connectivity_names = [
        dace_utils.connectivity_identifier(offset) for offset in neighbor_tables.keys()
    ]

    parent_sdfg = transformer.context.body
    parent_state = transformer.context.state

    input_mapping = {
        connector: dace.Memlet.from_array(node.data, node.desc(parent_sdfg))
        for connector, node in input_nodes.items()
    }
    connectivity_mapping = {
        name: dace.Memlet.from_array(name, parent_sdfg.arrays[name]) for name in connectivity_names
    }
    array_mapping = {**input_mapping, **connectivity_mapping}
    symbol_mapping = map_nested_sdfg_symbols(parent_sdfg, lift_context.body, array_mapping)

    nested_sdfg_node = parent_state.add_nested_sdfg(
        lift_context.body,
        parent_sdfg,
        inputs={*array_mapping.keys(), *iterator_index_nodes.keys()},
        outputs={inner_out_connector},
        symbol_mapping=symbol_mapping,
        debuginfo=lift_context.body.debuginfo,
    )

    for connectivity_connector, memlet in connectivity_mapping.items():
        parent_state.add_memlet_path(
            parent_state.add_access(memlet.data, debuginfo=lift_context.body.debuginfo),
            map_entry,
            nested_sdfg_node,
            dst_conn=connectivity_connector,
            memlet=memlet,
        )

    for inner_connector, access_node in input_nodes.items():
        parent_state.add_memlet_path(
            access_node,
            map_entry,
            nested_sdfg_node,
            dst_conn=inner_connector,
            memlet=input_mapping[inner_connector],
        )

    for inner_connector, access_node in iterator_index_nodes.items():
        memlet = dace.Memlet(data=access_node.data, subset="0")
        if inner_connector in lifted_index_connectors:
            parent_state.add_edge(access_node, None, nested_sdfg_node, inner_connector, memlet)
        else:
            parent_state.add_memlet_path(
                access_node, map_entry, nested_sdfg_node, dst_conn=inner_connector, memlet=memlet
            )

    parent_state.add_memlet_path(
        nested_sdfg_node,
        map_exit,
        neighbor_value_node,
        src_conn=inner_out_connector,
        memlet=dace.Memlet(data=neighbor_value_node.data, subset=",".join(map_entry.params)),
    )

    if offset_provider.has_skip_values:
        # check neighbor validity on if/else inter-state edge
        # use one branch for connectivity case
        start_state = lift_context.body.add_state_before(
            lift_context.body.start_state,
            "start",
            condition=f"{lifted_index_connectors[0]} != {neighbor_skip_value}",
        )
        # use the other branch for skip value case
        skip_neighbor_state = lift_context.body.add_state("skip_neighbor")
        skip_neighbor_state.add_edge(
            skip_neighbor_state.add_tasklet(
                "identity", {}, {"val"}, f"val = {transformer.context.reduce_identity.value}"
            ),
            "val",
            skip_neighbor_state.add_access(inner_outputs[0].value.data),
            None,
            dace.Memlet(data=inner_outputs[0].value.data, subset="0"),
        )
        lift_context.body.add_edge(
            start_state,
            skip_neighbor_state,
            dace.InterstateEdge(condition=f"{lifted_index_connectors[0]} == {neighbor_skip_value}"),
        )

    return [ValueExpr(neighbor_value_node, inner_outputs[0].dtype)]


def builtin_neighbors(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    sdfg: dace.SDFG = transformer.context.body
    state: dace.SDFGState = transformer.context.state

    di = dace_utils.debug_info(node, default=sdfg.debuginfo)
    offset_literal, data = node_args
    assert isinstance(offset_literal, itir.OffsetLiteral)
    offset_dim = offset_literal.value
    assert isinstance(offset_dim, str)
    offset_provider = transformer.offset_provider[offset_dim]
    if not isinstance(offset_provider, Connectivity):
        raise NotImplementedError(
            "Neighbor reduction only implemented for connectivity based on neighbor tables."
        )

    lift_node = None
    if isinstance(data, FunCall):
        assert isinstance(data.fun, itir.FunCall)
        fun_node = data.fun
        if isinstance(fun_node.fun, itir.SymRef) and fun_node.fun.id == "lift":
            lift_node = fun_node
            lift_args = transformer.visit(data.args)
            iterator = next(filter(lambda x: isinstance(x, IteratorExpr), lift_args), None)
    if lift_node is None:
        iterator = transformer.visit(data)
    assert isinstance(iterator, IteratorExpr)
    field_desc = iterator.field.desc(transformer.context.body)
    origin_index_node = iterator.indices[offset_provider.origin_axis.value]

    assert transformer.context.reduce_identity is not None
    assert transformer.context.reduce_identity.dtype == iterator.dtype

    # gather the neighbors in a result array dimensioned for `max_neighbors`
    neighbor_value_var = unique_var_name()
    sdfg.add_array(
        neighbor_value_var,
        dtype=iterator.dtype,
        shape=(offset_provider.max_neighbors,),
        transient=True,
    )
    neighbor_value_node = state.add_access(neighbor_value_var, debuginfo=di)

    # allocate scalar to store index for direct addressing of neighbor field
    neighbor_index_var = unique_var_name()
    sdfg.add_scalar(neighbor_index_var, _INDEX_DTYPE, transient=True)
    neighbor_index_node = state.add_access(neighbor_index_var, debuginfo=di)

    # generate unique map index name to avoid conflict with other maps inside same state
    neighbor_map_index = unique_name(f"{offset_dim}_neighbor_map_idx")
    me, mx = state.add_map(
        f"{offset_dim}_neighbor_map",
        ndrange={neighbor_map_index: f"0:{offset_provider.max_neighbors}"},
        debuginfo=di,
    )

    table_name = dace_utils.connectivity_identifier(offset_dim)
    shift_tasklet = state.add_tasklet(
        "shift",
        code=f"__result = __table[__idx, {neighbor_map_index}]",
        inputs={"__table", "__idx"},
        outputs={"__result"},
        debuginfo=di,
    )
    state.add_memlet_path(
        state.add_access(table_name, debuginfo=di),
        me,
        shift_tasklet,
        memlet=dace.Memlet.from_array(table_name, sdfg.arrays[table_name]),
        dst_conn="__table",
    )
    state.add_memlet_path(
        origin_index_node,
        me,
        shift_tasklet,
        memlet=dace.Memlet(data=origin_index_node.data, subset="0"),
        dst_conn="__idx",
    )
    state.add_edge(
        shift_tasklet,
        "__result",
        neighbor_index_node,
        None,
        dace.Memlet(data=neighbor_index_var, subset="0"),
    )

    if lift_node is not None:
        _visit_lift_in_neighbors_reduction(
            transformer,
            lift_node,
            lift_args,
            offset_provider,
            me,
            mx,
            neighbor_index_node,
            neighbor_value_node,
        )
    else:
        sorted_dims = transformer.get_sorted_field_dimensions(iterator.dimensions)
        data_access_index = ",".join(f"{dim}_v" for dim in sorted_dims)
        connector_neighbor_dim = f"{offset_provider.neighbor_axis.value}_v"
        data_access_tasklet = state.add_tasklet(
            "data_access",
            code=f"__data = __field[{data_access_index}] "
            + (
                f"if {connector_neighbor_dim} != {neighbor_skip_value} else {transformer.context.reduce_identity.value}"
                if offset_provider.has_skip_values
                else ""
            ),
            inputs={"__field"} | {f"{dim}_v" for dim in iterator.dimensions},
            outputs={"__data"},
            debuginfo=di,
        )
        state.add_memlet_path(
            iterator.field,
            me,
            data_access_tasklet,
            memlet=dace.Memlet.from_array(iterator.field.data, field_desc),
            dst_conn="__field",
        )
        for dim in iterator.dimensions:
            connector = f"{dim}_v"
            if dim == offset_provider.neighbor_axis.value:
                state.add_edge(
                    neighbor_index_node,
                    None,
                    data_access_tasklet,
                    connector,
                    dace.Memlet(data=neighbor_index_var, subset="0"),
                )
            else:
                state.add_memlet_path(
                    iterator.indices[dim],
                    me,
                    data_access_tasklet,
                    dst_conn=connector,
                    memlet=dace.Memlet(data=iterator.indices[dim].data, subset="0"),
                )

        state.add_memlet_path(
            data_access_tasklet,
            mx,
            neighbor_value_node,
            memlet=dace.Memlet(data=neighbor_value_var, subset=neighbor_map_index),
            src_conn="__data",
        )

    if not offset_provider.has_skip_values:
        return [ValueExpr(neighbor_value_node, iterator.dtype)]
    else:
        """
        In case of neighbor tables with skip values, in addition to the array of neighbor values this function also
        returns an array of booleans to indicate if the neighbor value is present or not. This node is only used
        for neighbor reductions with lambda functions, a very specific case. For single input neighbor reductions,
        the regular case, this node will be removed by the simplify pass.
        """
        neighbor_valid_var = unique_var_name()
        sdfg.add_array(
            neighbor_valid_var,
            dtype=dace.dtypes.bool,
            shape=(offset_provider.max_neighbors,),
            transient=True,
        )
        neighbor_valid_node = state.add_access(neighbor_valid_var, debuginfo=di)

        neighbor_valid_tasklet = state.add_tasklet(
            f"check_valid_neighbor_{offset_dim}",
            {"__idx"},
            {"__valid"},
            f"__valid = True if __idx != {neighbor_skip_value} else False",
            debuginfo=di,
        )
        state.add_edge(
            neighbor_index_node,
            None,
            neighbor_valid_tasklet,
            "__idx",
            dace.Memlet(data=neighbor_index_var, subset="0"),
        )
        state.add_memlet_path(
            neighbor_valid_tasklet,
            mx,
            neighbor_valid_node,
            memlet=dace.Memlet(data=neighbor_valid_var, subset=neighbor_map_index),
            src_conn="__valid",
        )
        return [
            ValueExpr(neighbor_value_node, iterator.dtype),
            ValueExpr(neighbor_valid_node, dace.dtypes.bool),
        ]


def builtin_can_deref(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_utils.debug_info(node, default=transformer.context.body.debuginfo)
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
            dace.Memlet(data=result_name, subset="0"),
        )
        return [ValueExpr(result_node, dace.dtypes.bool)]

    # create tasklet to check that field indices are non-negative (-1 is invalid)
    args = [ValueExpr(access_node, _INDEX_DTYPE) for access_node in iterator.indices.values()]
    internals = [f"{arg.value.data}_v" for arg in args]
    expr_code = " and ".join(f"{v} != {neighbor_skip_value}" for v in internals)

    return transformer.add_expr_tasklet(
        list(zip(args, internals)), expr_code, dace.dtypes.bool, "can_deref", dace_debuginfo=di
    )


def builtin_if(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    assert len(node_args) == 3
    sdfg = transformer.context.body
    current_state = transformer.context.state
    is_start_state = sdfg.start_block == current_state

    # build an empty state to join true and false branches
    join_state = sdfg.add_state_before(current_state, "join")

    def build_if_state(arg, state):
        symbol_map = copy.deepcopy(transformer.context.symbol_map)
        node_context = Context(sdfg, state, symbol_map)
        node_taskgen = PythonTaskletCodegen(
            transformer.offset_provider,
            node_context,
            transformer.use_field_canonical_representation,
        )
        return node_taskgen.visit(arg)

    # represent the if-statement condition as a tasklet inside an `if_statement` state preceding `join` state
    stmt_state = sdfg.add_state_before(join_state, "if_statement", is_start_state)
    stmt_node = build_if_state(node_args[0], stmt_state)[0]
    assert isinstance(stmt_node, ValueExpr)
    assert stmt_node.dtype == dace.dtypes.bool
    assert sdfg.arrays[stmt_node.value.data].shape == (1,)

    # visit true and false branches (here called `tbr` and `fbr`) as separate states, following `if_statement` state
    tbr_state = sdfg.add_state("true_branch")
    sdfg.add_edge(
        stmt_state, tbr_state, dace.InterstateEdge(condition=f"{stmt_node.value.data} == True")
    )
    sdfg.add_edge(tbr_state, join_state, dace.InterstateEdge())
    tbr_values = flatten_list(build_if_state(node_args[1], tbr_state))
    #
    fbr_state = sdfg.add_state("false_branch")
    sdfg.add_edge(
        stmt_state, fbr_state, dace.InterstateEdge(condition=f"{stmt_node.value.data} == False")
    )
    sdfg.add_edge(fbr_state, join_state, dace.InterstateEdge())
    fbr_values = flatten_list(build_if_state(node_args[2], fbr_state))

    assert isinstance(stmt_node, ValueExpr)
    assert stmt_node.dtype == dace.dtypes.bool
    # make the result of the if-statement evaluation available inside current state
    ctx_stmt_node = ValueExpr(current_state.add_access(stmt_node.value.data), stmt_node.dtype)

    # we distinguish between select if-statements, where both true and false branches are symbolic expressions,
    # and therefore do not require exclusive branch execution, and regular if-statements where at least one branch
    # is a value expression, which has to be evaluated at runtime with conditional state transition
    result_values = []
    assert len(tbr_values) == len(fbr_values)
    for tbr_value, fbr_value in zip(tbr_values, fbr_values):
        assert isinstance(tbr_value, (SymbolExpr, ValueExpr))
        assert isinstance(fbr_value, (SymbolExpr, ValueExpr))
        assert tbr_value.dtype == fbr_value.dtype

        if all(isinstance(x, SymbolExpr) for x in (tbr_value, fbr_value)):
            # both branches return symbolic expressions, therefore the if-node can be translated
            # to a select-tasklet inside current state
            # TODO: use select-memlet when it becomes available in dace
            code = f"{tbr_value.value} if _cond else {fbr_value.value}"
            if_expr = transformer.add_expr_tasklet(
                [(ctx_stmt_node, "_cond")], code, tbr_value.dtype, "if_select"
            )[0]
            result_values.append(if_expr)
        else:
            # at least one of the two branches contains a value expression, which should be evaluated
            # only if the corresponding true/false condition is satisfied
            desc = sdfg.arrays[
                tbr_value.value.data if isinstance(tbr_value, ValueExpr) else fbr_value.value.data
            ]
            var = unique_var_name()
            if isinstance(desc, dace.data.Scalar):
                sdfg.add_scalar(var, desc.dtype, transient=True)
            else:
                sdfg.add_array(var, desc.shape, desc.dtype, transient=True)

            # write result to transient data container and access it in the original state
            for state, expr in [(tbr_state, tbr_value), (fbr_state, fbr_value)]:
                val_node = state.add_access(var)
                if isinstance(expr, ValueExpr):
                    state.add_nedge(
                        expr.value, val_node, dace.Memlet.from_array(expr.value.data, desc)
                    )
                else:
                    assert desc.shape == (1,)
                    state.add_edge(
                        state.add_tasklet("write_symbol", {}, {"_out"}, f"_out = {expr.value}"),
                        "_out",
                        val_node,
                        None,
                        dace.Memlet(var, "0"),
                    )
            result_values.append(ValueExpr(current_state.add_access(var), desc.dtype))

    if tbr_state.is_empty() and fbr_state.is_empty():
        # if all branches are symbolic expressions, the true/false and join states can be removed
        # as well as the conditional state transition
        sdfg.remove_nodes_from([join_state, tbr_state, fbr_state])
        sdfg.add_edge(stmt_state, current_state, dace.InterstateEdge())
    elif tbr_state.is_empty():
        # use direct edge from if-statement to join state for true branch
        tbr_condition = sdfg.edges_between(stmt_state, tbr_state)[0].condition
        sdfg.edges_between(stmt_state, join_state)[0].contition = tbr_condition
        sdfg.remove_node(tbr_state)
    elif fbr_state.is_empty():
        # use direct edge from if-statement to join state for false branch
        fbr_condition = sdfg.edges_between(stmt_state, fbr_state)[0].condition
        sdfg.edges_between(stmt_state, join_state)[0].contition = fbr_condition
        sdfg.remove_node(fbr_state)
    else:
        # remove direct edge from if-statement to join state
        sdfg.remove_edge(sdfg.edges_between(stmt_state, join_state)[0])
        # the if-statement condition is not used in current state
        current_state.remove_node(ctx_stmt_node.value)

    return result_values


def builtin_list_get(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_utils.debug_info(node, default=transformer.context.body.debuginfo)
    args = list(itertools.chain(*transformer.visit(node_args)))
    assert len(args) == 2
    # index node
    if isinstance(args[0], SymbolExpr):
        index_value = args[0].value
        result_name = unique_var_name()
        transformer.context.body.add_scalar(result_name, args[1].dtype, transient=True)
        result_node = transformer.context.state.add_access(result_name)
        transformer.context.state.add_nedge(
            args[1].value, result_node, dace.Memlet(data=args[1].value.data, subset=index_value)
        )
        return [ValueExpr(result_node, args[1].dtype)]

    else:
        expr_args = [(arg, f"{arg.value.data}_v") for arg in args]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[1]}[{internals[0]}]"
        return transformer.add_expr_tasklet(
            expr_args, expr, args[1].dtype, "list_get", dace_debuginfo=di
        )


def builtin_cast(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_utils.debug_info(node, default=transformer.context.body.debuginfo)
    args = transformer.visit(node_args[0])
    internals = [f"{arg.value.data}_v" for arg in args]
    target_type = node_args[1]
    assert isinstance(target_type, itir.SymRef)
    expr = _MATH_BUILTINS_MAPPING[target_type.id].format(*internals)
    type_ = itir_type_as_dace_type(node.type)  # type: ignore[arg-type]  # ensure by type inference
    return transformer.add_expr_tasklet(
        list(zip(args, internals)), expr, type_, "cast", dace_debuginfo=di
    )


def builtin_make_const_list(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    di = dace_utils.debug_info(node, default=transformer.context.body.debuginfo)
    args = [transformer.visit(arg)[0] for arg in node_args]
    assert all(isinstance(x, (SymbolExpr, ValueExpr)) for x in args)
    args_dtype = [x.dtype for x in args]
    assert len(set(args_dtype)) == 1
    dtype = args_dtype[0]

    var_name = unique_var_name()
    transformer.context.body.add_array(var_name, (len(args),), dtype, transient=True)
    var_node = transformer.context.state.add_access(var_name, debuginfo=di)

    for i, arg in enumerate(args):
        if isinstance(arg, SymbolExpr):
            transformer.context.state.add_edge(
                transformer.context.state.add_tasklet(
                    f"get_arg{i}", {}, {"val"}, f"val = {arg.value}"
                ),
                "val",
                var_node,
                None,
                dace.Memlet(data=var_name, subset=f"{i}"),
            )
        else:
            assert arg.value.desc(transformer.context.body).shape == (1,)
            transformer.context.state.add_nedge(
                arg.value,
                var_node,
                dace.Memlet(data=arg.value.data, subset="0", other_subset=f"{i}"),
            )

    return [ValueExpr(var_node, dtype)]


def builtin_make_tuple(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [transformer.visit(arg) for arg in node_args]
    return args


def builtin_tuple_get(
    transformer: PythonTaskletCodegen, node: itir.Expr, node_args: list[itir.Expr]
) -> list[ValueExpr]:
    elements = transformer.visit(node_args[1])
    index = node_args[0]
    if isinstance(index, itir.Literal):
        return [elements[int(index.value)]]
    raise ValueError("Tuple can only be subscripted with compile-time constants.")


_GENERAL_BUILTIN_MAPPING: dict[
    str, Callable[[PythonTaskletCodegen, itir.Expr, list[itir.Expr]], list[ValueExpr]]
] = {
    "can_deref": builtin_can_deref,
    "cast_": builtin_cast,
    "if_": builtin_if,
    "list_get": builtin_list_get,
    "make_const_list": builtin_make_const_list,
    "make_tuple": builtin_make_tuple,
    "neighbors": builtin_neighbors,
    "tuple_get": builtin_tuple_get,
}


class GatherLambdaSymbolsPass(eve.NodeVisitor):
    _sdfg: dace.SDFG
    _state: dace.SDFGState
    _symbol_map: dict[str, TaskletExpr | tuple[ValueExpr]]
    _parent_symbol_map: dict[str, TaskletExpr]

    def __init__(self, sdfg, state, parent_symbol_map):
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
            # update table of lambda symbols
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
            # update table of lambda symbols
            field = self._state.add_access(param, debuginfo=self._sdfg.debuginfo)
            indices = {
                dim: self._state.add_access(index_arg, debuginfo=self._sdfg.debuginfo)
                for dim, index_arg in index_names.items()
            }
            self._symbol_map[param] = IteratorExpr(field, indices, arg.dtype, arg.dimensions)
        else:
            assert isinstance(arg, SymbolExpr)
            self._symbol_map[param] = arg

    def _add_tuple(self, param, args):
        nodes = []
        # create storage in lambda sdfg for each tuple element
        for arg in args:
            var = unique_var_name()
            self._sdfg.add_scalar(var, dtype=arg.dtype)
            arg_node = self._state.add_access(var, debuginfo=self._sdfg.debuginfo)
            nodes.append(ValueExpr(arg_node, arg.dtype))
        # update table of lambda symbols
        self._symbol_map[param] = tuple(nodes)

    def visit_SymRef(self, node: itir.SymRef):
        name = str(node.id)
        if name in self._parent_symbol_map and name not in self._symbol_map:
            arg = self._parent_symbol_map[name]
            self._add_symbol(name, arg)

    def visit_Lambda(self, node: itir.Lambda, args: Optional[Sequence[TaskletExpr]] = None):
        if args is not None:
            if len(node.params) == len(args):
                for param, arg in zip(node.params, args):
                    self._add_symbol(str(param.id), arg)
            else:
                # implicitly make tuple
                assert len(node.params) == 1
                self._add_tuple(str(node.params[0].id), args)
        self.visit(node.expr)


class GatherOutputSymbolsPass(eve.NodeVisitor):
    _sdfg: dace.SDFG
    _state: dace.SDFGState
    _symbol_map: dict[str, TaskletExpr]

    @property
    def symbol_refs(self):
        """Dictionary of symbols referenced from the output expression."""
        return self._symbol_map

    def __init__(self, sdfg, state):
        self._sdfg = sdfg
        self._state = state
        self._symbol_map = {}

    def visit_SymRef(self, node: itir.SymRef):
        param = str(node.id)
        if param not in _GENERAL_BUILTIN_MAPPING and param not in self._symbol_map:
            access_node = self._state.add_access(param, debuginfo=self._sdfg.debuginfo)
            self._symbol_map[param] = ValueExpr(
                access_node,
                dtype=itir_type_as_dace_type(node.type),  # type: ignore[arg-type]  # ensure by type inference
            )


class PythonTaskletCodegen(gt4py.eve.codegen.TemplatedGenerator):
    offset_provider: dict[str, Any]
    context: Context
    use_field_canonical_representation: bool

    def __init__(
        self,
        offset_provider: dict[str, Any],
        context: Context,
        use_field_canonical_representation: bool,
    ):
        self.offset_provider = offset_provider
        self.context = context
        self.use_field_canonical_representation = use_field_canonical_representation

    def get_sorted_field_dimensions(self, dims: Sequence[str]):
        return sorted(dims) if self.use_field_canonical_representation else dims

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
            get_used_connectivities(node, self.offset_provider) if use_neighbor_tables else {}
        )
        connectivity_names = [
            dace_utils.connectivity_identifier(offset) for offset in neighbor_tables.keys()
        ]

        # Create the SDFG for the lambda's body
        lambda_sdfg = dace.SDFG(func_name)
        lambda_sdfg.debuginfo = dace_utils.debug_info(node, default=self.context.body.debuginfo)
        lambda_state = lambda_sdfg.add_state(f"{func_name}_body", is_start_block=True)

        lambda_symbols_pass = GatherLambdaSymbolsPass(
            lambda_sdfg, lambda_state, self.context.symbol_map
        )
        lambda_symbols_pass.visit(node, args=args)

        # Add for input nodes for lambda symbols
        inputs: list[tuple[str, ValueExpr] | tuple[tuple[str, dict], IteratorExpr]] = []
        for sym, input_node in lambda_symbols_pass.symbol_refs.items():
            params = [str(p.id) for p in node.params]
            try:
                param_index = params.index(sym)
            except ValueError:
                param_index = -1
            if param_index >= 0:
                outer_node = args[param_index]
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
            elif isinstance(input_node, tuple):
                assert param_index >= 0
                for i, input_node_i in enumerate(input_node):
                    arg_i = args[param_index + i]
                    assert isinstance(arg_i, ValueExpr)
                    assert isinstance(input_node_i, ValueExpr)
                    inputs.append((input_node_i.value.data, arg_i))

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
        lambda_taskgen = PythonTaskletCodegen(
            self.offset_provider,
            lambda_context,
            self.use_field_canonical_representation,
        )

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
                    expr.value, result_access, dace.Memlet(data=result_access.data, subset="0")
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
        return [SymbolExpr(node.value, itir_type_as_dace_type(node.type))]

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
                nsdfg_inputs[name] = dace.Memlet.from_array(
                    value.value.data, self.context.body.arrays[value.value.data]
                )
            else:
                assert isinstance(value, IteratorExpr)
                field = name[0]
                indices = name[1]
                nsdfg_inputs[field] = dace.Memlet.from_array(
                    value.field.data, self.context.body.arrays[value.field.data]
                )
                for dim, var in indices.items():
                    store = value.indices[dim].data
                    nsdfg_inputs[var] = dace.Memlet.from_array(
                        store, self.context.body.arrays[store]
                    )

        neighbor_tables = get_used_connectivities(node.fun, self.offset_provider)
        for offset in neighbor_tables.keys():
            var = dace_utils.connectivity_identifier(offset)
            nsdfg_inputs[var] = dace.Memlet.from_array(var, self.context.body.arrays[var])

        symbol_mapping = map_nested_sdfg_symbols(self.context.body, func_context.body, nsdfg_inputs)

        nsdfg_node = self.context.state.add_nested_sdfg(
            func_context.body,
            None,
            inputs=set(nsdfg_inputs.keys()),
            outputs=set(r.value.data for r in results),
            symbol_mapping=symbol_mapping,
            debuginfo=dace_utils.debug_info(node, default=func_context.body.debuginfo),
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
        for offset in neighbor_tables.keys():
            var = dace_utils.connectivity_identifier(offset)
            memlet = nsdfg_inputs[var]
            access = self.context.state.add_access(var, debuginfo=nsdfg_node.debuginfo)
            self.context.state.add_edge(access, None, nsdfg_node, var, memlet)

        result_exprs = []
        for result in results:
            name = unique_var_name()
            self.context.body.add_scalar(name, result.dtype, transient=True)
            result_access = self.context.state.add_access(name, debuginfo=nsdfg_node.debuginfo)
            result_exprs.append(ValueExpr(result_access, result.dtype))
            memlet = dace.Memlet.from_array(name, self.context.body.arrays[name])
            self.context.state.add_edge(nsdfg_node, result.value.data, result_access, None, memlet)

        return result_exprs

    def _visit_deref(self, node: itir.FunCall) -> list[ValueExpr]:
        di = dace_utils.debug_info(node, default=self.context.body.debuginfo)
        iterator = self.visit(node.args[0])
        if not isinstance(iterator, IteratorExpr):
            # already a list of ValueExpr
            return iterator

        sorted_dims = self.get_sorted_field_dimensions(iterator.dimensions)
        if all([dim in iterator.indices for dim in iterator.dimensions]):
            # The deref iterator has index values on all dimensions: the result will be a scalar
            args = [ValueExpr(iterator.field, iterator.dtype)] + [
                ValueExpr(iterator.indices[dim], _INDEX_DTYPE) for dim in sorted_dims
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{', '.join(internals[1:])}]"
            return self.add_expr_tasklet(
                list(zip(args, internals)), expr, iterator.dtype, "deref", dace_debuginfo=di
            )

        else:
            dims_not_indexed = [dim for dim in iterator.dimensions if dim not in iterator.indices]
            assert len(dims_not_indexed) == 1
            offset = dims_not_indexed[0]
            offset_provider = self.offset_provider[offset]
            neighbor_dim = offset_provider.neighbor_axis.value

            result_name = unique_var_name()
            self.context.body.add_array(
                result_name, (offset_provider.max_neighbors,), iterator.dtype, transient=True
            )
            result_array = self.context.body.arrays[result_name]
            result_node = self.context.state.add_access(result_name, debuginfo=di)

            deref_connectors = ["_inp"] + [
                f"_i_{dim}" for dim in sorted_dims if dim in iterator.indices
            ]
            deref_nodes = [iterator.field] + [
                iterator.indices[dim] for dim in sorted_dims if dim in iterator.indices
            ]
            deref_memlets = [
                dace.Memlet.from_array(iterator.field.data, iterator.field.desc(self.context.body))
            ] + [dace.Memlet(data=node.data, subset="0") for node in deref_nodes[1:]]

            # we create a mapped tasklet for array slicing
            index_name = unique_name(f"_i_{neighbor_dim}")
            map_ranges = {index_name: f"0:{offset_provider.max_neighbors}"}
            src_subset = ",".join(
                [f"_i_{dim}" if dim in iterator.indices else index_name for dim in sorted_dims]
            )
            self.context.state.add_mapped_tasklet(
                "deref",
                map_ranges,
                inputs={k: v for k, v in zip(deref_connectors, deref_memlets)},
                outputs={"_out": dace.Memlet.from_array(result_name, result_array)},
                code=f"_out[{index_name}] = _inp[{src_subset}]",
                external_edges=True,
                input_nodes={node.data: node for node in deref_nodes},
                output_nodes={result_name: result_node},
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
        di = dace_utils.debug_info(node, default=self.context.body.debuginfo)
        shift = node.fun
        assert isinstance(shift, itir.FunCall)
        tail, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0])
        if not isinstance(iterator, IteratorExpr):
            # shift cannot be applied because the argument is not iterable
            # TODO: remove this special case when ITIR pass is able to catch it
            assert isinstance(iterator, list) and len(iterator) == 1
            assert isinstance(iterator[0], ValueExpr)
            return iterator

        assert isinstance(tail[0], itir.OffsetLiteral)
        offset_dim = tail[0].value
        assert isinstance(offset_dim, str)
        offset_node = self.visit(tail[1])[0]
        assert offset_node.dtype in dace.dtypes.INTEGER_TYPES

        if isinstance(self.offset_provider[offset_dim], Connectivity):
            offset_provider = self.offset_provider[offset_dim]
            connectivity = self.context.state.add_access(
                dace_utils.connectivity_identifier(offset_dim), debuginfo=di
            )

            shifted_dim = offset_provider.origin_axis.value
            target_dim = offset_provider.neighbor_axis.value
            args = [
                ValueExpr(connectivity, _INDEX_DTYPE),
                ValueExpr(iterator.indices[shifted_dim], offset_node.dtype),
                offset_node,
            ]
            internals = [f"{arg.value.data}_v" for arg in args]
            expr = f"{internals[0]}[{internals[1]}, {internals[2]}]"
        else:
            assert isinstance(self.offset_provider[offset_dim], Dimension)

            shifted_dim = self.offset_provider[offset_dim].value
            target_dim = shifted_dim
            args = [ValueExpr(iterator.indices[shifted_dim], offset_node.dtype), offset_node]
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
        di = dace_utils.debug_info(node, default=self.context.body.debuginfo)
        offset = node.value
        assert isinstance(offset, int)
        offset_var = unique_var_name()
        self.context.body.add_scalar(offset_var, _INDEX_DTYPE, transient=True)
        offset_node = self.context.state.add_access(offset_var, debuginfo=di)
        tasklet_node = self.context.state.add_tasklet(
            "get_offset", {}, {"__out"}, f"__out = {offset}", debuginfo=di
        )
        self.context.state.add_edge(
            tasklet_node, "__out", offset_node, None, dace.Memlet(data=offset_var, subset="0")
        )
        return [ValueExpr(offset_node, self.context.body.arrays[offset_var].dtype)]

    def _visit_reduce(self, node: itir.FunCall):
        di = dace_utils.debug_info(node, default=self.context.body.debuginfo)
        reduce_dtype = itir_type_as_dace_type(node.type)  # type: ignore[arg-type]  # ensure by type inference

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

            args = self.visit(node.args[0])

            assert 1 <= len(args) <= 2
            reduce_input_node = args[0].value

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

            args = self.visit(node.args)

            # clear context
            self.context.reduce_identity = None

            # check that all neighbor expressions have the same shape
            args_shape = [
                arg[0].value.desc(self.context.body).shape
                for arg in args
                if arg[0].value.desc(self.context.body).shape != (1,)
            ]
            assert len(set(args_shape)) == 1
            nreduce_shape = args_shape[0]

            input_args = [arg[0] for arg in args]
            input_valid_args = [arg[1] for arg in args if len(arg) == 2]

            assert len(nreduce_shape) == 1
            nreduce_index = unique_name("_i")
            nreduce_domain = {nreduce_index: f"0:{nreduce_shape[0]}"}

            reduce_input_name = unique_var_name()
            self.context.body.add_array(
                reduce_input_name, nreduce_shape, reduce_dtype, transient=True
            )

            lambda_node = itir.Lambda(
                expr=fun_node.expr.args[1], params=fun_node.params[1:], location=node.location
            )
            lambda_context, inner_inputs, inner_outputs = self.visit(
                lambda_node, args=input_args, use_neighbor_tables=False
            )

            input_mapping = {
                param: (
                    dace.Memlet(data=arg.value.data, subset="0")
                    if arg.value.desc(self.context.body).shape == (1,)
                    else dace.Memlet(data=arg.value.data, subset=nreduce_index)
                )
                for (param, _), arg in zip(inner_inputs, input_args)
            }
            output_mapping = {
                inner_outputs[0].value.data: dace.Memlet(
                    data=reduce_input_name, subset=nreduce_index
                )
            }
            symbol_mapping = map_nested_sdfg_symbols(
                self.context.body, lambda_context.body, input_mapping
            )

            if input_valid_args:
                """
                The neighbors builtin returns an array of booleans in case the connectivity table contains skip values.
                These booleans indicate whether the neighbor is present or not, and are used in a tasklet to select
                the result of field access or the identity value, respectively.
                If the neighbor table has full connectivity (no skip values by type definition), the input_valid node
                is not built, and the construction of the select tasklet below is also skipped.
                """
                input_args.append(input_valid_args[0])
                input_valid_node = input_valid_args[0].value
                lambda_output_node = inner_outputs[0].value
                # add input connector to nested sdfg
                lambda_context.body.add_scalar("_valid_neighbor", dace.dtypes.bool)
                input_mapping["_valid_neighbor"] = dace.Memlet(
                    data=input_valid_node.data, subset=nreduce_index
                )
                # add select tasklet before writing to output node
                # TODO: consider replacing it with a select-memlet once it is supported by DaCe SDFG API
                output_edge = lambda_context.state.in_edges(lambda_output_node)[0]
                assert isinstance(
                    lambda_context.body.arrays[output_edge.src.data], dace.data.Scalar
                )
                select_tasklet = lambda_context.state.add_tasklet(
                    "neighbor_select",
                    {"_inp", "_valid"},
                    {"_out"},
                    f"_out = _inp if _valid else {reduce_identity}",
                )
                lambda_context.state.add_edge(
                    output_edge.src,
                    None,
                    select_tasklet,
                    "_inp",
                    dace.Memlet(data=output_edge.src.data, subset="0"),
                )
                lambda_context.state.add_edge(
                    lambda_context.state.add_access("_valid_neighbor"),
                    None,
                    select_tasklet,
                    "_valid",
                    dace.Memlet(data="_valid_neighbor", subset="0"),
                )
                lambda_context.state.add_edge(
                    select_tasklet,
                    "_out",
                    lambda_output_node,
                    None,
                    dace.Memlet(data=lambda_output_node.data, subset="0"),
                )
                lambda_context.state.remove_edge(output_edge)

            reduce_input_node = self.context.state.add_access(reduce_input_name, debuginfo=di)

            nsdfg_node, map_entry, _ = add_mapped_nested_sdfg(
                self.context.state,
                sdfg=lambda_context.body,
                map_ranges=nreduce_domain,
                inputs=input_mapping,
                outputs=output_mapping,
                symbol_mapping=symbol_mapping,
                input_nodes={arg.value.data: arg.value for arg in input_args},
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
            reduce_node, result_access, dace.Memlet(data=result_name, subset="0")
        )

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
        type_ = itir_type_as_dace_type(node.type)  # type: ignore[arg-type]  # ensure by type inference
        return self.add_expr_tasklet(
            expr_args,
            expr,
            type_,
            "numeric",
            dace_debuginfo=dace_utils.debug_info(node, default=self.context.body.debuginfo),
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
                memlet = dace.Memlet.from_array(
                    arg.value.data, self.context.body.arrays[arg.value.data]
                )
                self.context.state.add_edge(arg.value, None, expr_tasklet, internal, memlet)

        memlet = dace.Memlet(data=result_access.data, subset="0")
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
    use_field_canonical_representation: bool,
) -> tuple[Context, Sequence[ValueExpr]]:
    body = dace.SDFG("tasklet_toplevel")
    body.debuginfo = dace_utils.debug_info(node)
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
        state.add_edge(tasklet, "value", access, None, dace.Memlet(data=name, subset="0"))
    for name, ty in inputs:
        if isinstance(ty, ts.FieldType):
            ndim = len(ty.dims)
            shape, strides = new_array_symbols(name, ndim)
            dims = [dim.value for dim in ty.dims]
            dtype = dace_utils.as_dace_type(ty.dtype)
            body.add_array(name, shape=shape, strides=strides, dtype=dtype)
            field = state.add_access(name, debuginfo=body.debuginfo)
            indices = {dim: idx_accesses[dim] for dim in domain.keys()}
            symbol_map[name] = IteratorExpr(field, indices, dtype, dims)
        else:
            assert isinstance(ty, ts.ScalarType)
            dtype = dace_utils.as_dace_type(ty)
            body.add_scalar(name, dtype=dtype)
            symbol_map[name] = ValueExpr(state.add_access(name, debuginfo=body.debuginfo), dtype)
    for arr, name in connectivities:
        shape, strides = new_array_symbols(name, ndim=2)
        body.add_array(name, shape=shape, strides=strides, dtype=arr.dtype)

    context = Context(body, state, symbol_map)
    translator = PythonTaskletCodegen(offset_provider, context, use_field_canonical_representation)

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
