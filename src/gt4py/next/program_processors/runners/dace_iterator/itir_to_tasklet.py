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
from typing import Any, Callable, Optional

import dace
import numpy as np

import gt4py.eve.codegen
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from .utility import connectivity_identifier, create_memlet_at, create_memlet_full


_TYPE_MAPPING = {
    "float": dace.float64,
    "float32": dace.float32,
    "float64": dace.float64,
    "int": dace.int32 if np.dtype(int).itemsize == 4 else dace.int64,
    "int32": dace.int32,
    "int64": dace.int64,
    "bool": dace.bool_,
}

_MATH_BUILTINS_MAPPING = {
    "abs": "abs({})",
    "sin": "math.sin({})",
    "cos": "math.cos({})",
    "tan": "math.tan({})",
    "arcsin": "math.asin({})",
    "arccos": "math.acos({})",
    "arctan": "math.atan({})",
    "sinh": "math.sinh({})",
    "cosh": "math.cosh({})",
    "tanh": "math.tanh({})",
    "arcsinh": "math.asinh({})",
    "arccosh": "math.acosh({})",
    "arctanh": "math.atanh({})",
    "sqrt": "math.sqrt({})",
    "exp": "math.exp({})",
    "log": "math.log({})",
    "gamma": "math.gamma({})",
    "cbrt": "math.cbrt({})",
    "isfinite": "math.isfinite({})",
    "isinf": "math.isinf({})",
    "isnan": "math.isnan({})",
    "floor": "math.floor({})",
    "ceil": "math.ceil({})",
    "trunc": "math.trunc({})",
    "minimum": "min({}, {})",
    "maximum": "max({}, {})",
    "fmod": "math.fmod({}, {})",
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
    "not_": "~{}",
}


@dataclasses.dataclass
class ClosureArg:
    memlet: dace.Memlet
    strides: tuple[dace.symbol]
    dtype: dace.dtypes.typeclass


@dataclasses.dataclass
class ValueExpr:
    value: dace.nodes.AccessNode


@dataclasses.dataclass
class IteratorExpr:
    field: dace.nodes.AccessNode
    indices: dict[str, dace.nodes.AccessNode]


def builtin_if(transformer: "PythonTaskletCodegen", node_args: list[itir.Expr]) -> list[ValueExpr]:
    args: list[dace.nodes.AccessNode] = transformer.visit(node_args)
    internals = [f"{arg.data}_v" for arg in args]
    expr = "({1} if {0} else {2})".format(*internals)
    return transformer.add_expr_tasklet(list(zip(args, internals)), expr, dace.dtypes.float64, "if")


def builtin_cast(
    transformer: "PythonTaskletCodegen", node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [transformer.visit(node_args[0])[0]]
    internals = [f"{arg.value.data}_v" for arg in args]
    target_type = node_args[1]
    assert isinstance(target_type, itir.SymRef)
    expr = _MATH_BUILTINS_MAPPING[target_type.id].format(*internals)
    return transformer.add_expr_tasklet(
        list(zip(args, internals)), expr, dace.dtypes.float64, "cast"
    )


def builtin_make_tuple(
    transformer: "PythonTaskletCodegen", node_args: list[itir.Expr]
) -> list[ValueExpr]:
    args = [transformer.visit(arg) for arg in node_args]
    return args


def builtin_tuple_get(
    transformer: "PythonTaskletCodegen", node_args: list[itir.Expr]
) -> list[ValueExpr]:
    elements = transformer.visit(node_args[1])
    index = node_args[0]
    if isinstance(index, itir.Literal):
        return elements[int(index.value)]
    raise ValueError("Tuple can only be subscripted with compile-time constants")


def builtin_undefined(*args: Any) -> Any:
    raise NotImplementedError()


_GENERAL_BUILTIN_MAPPING: dict[
    str, Callable[["PythonTaskletCodegen", list[itir.Expr]], list[ValueExpr]]
] = {
    "make_tuple": builtin_make_tuple,
    "tuple_get": builtin_tuple_get,
    "if_": builtin_if,
    "cast_": builtin_cast,
}


class PythonTaskletCodegen(gt4py.eve.codegen.TemplatedGenerator):
    sdfg: Optional[dace.SDFG]
    entry_state: Optional[dace.SDFGState]
    offset_provider: dict[str, Any]
    domain: dict[str, str]
    input_args: Sequence[ClosureArg]
    output_args: Sequence[ClosureArg]
    conn_args: Sequence[ClosureArg]
    params: list[str]
    results: list[str]

    def __init__(
        self,
        offset_provider: dict[str, Any],
        domain: dict[str, str],
        input_args: Sequence[ClosureArg],
        output_args: Sequence[ClosureArg],
        conn_args: Sequence[ClosureArg],
    ):
        self.sdfg = None
        self.entry_state = None
        self.offset_provider = offset_provider
        self.domain = domain
        self.input_args = input_args
        self.output_args = output_args
        self.conn_args = conn_args

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition, **kwargs):
        raise NotImplementedError()

    def visit_Lambda(self, node: itir.Lambda, **kwargs):
        assert not self.sdfg

        func_name = f"lambda_{abs(hash(node)):x}"
        param_names = [str(p.id) for p in node.params]
        result_names = [f"__result_{str(i)}" for i in range(len(self.output_args))]
        conn_names = [conn.memlet.data for conn in self.conn_args]

        self.params = param_names
        self.results = result_names

        # Create the SDFG for the function's body
        self.sdfg = dace.SDFG(func_name)
        self.entry_state = self.sdfg.add_state(f"{func_name}_entry", True)

        # Add input, connectivity, and output parameters as arrays
        array_names = [*param_names, *result_names, *conn_names]
        args = [*self.input_args, *self.output_args, *self.conn_args]
        for name, arg in zip(array_names, args):
            shape = arg.memlet.subset.size()
            self.sdfg.add_array(name, shape=shape, strides=arg.strides, dtype=arg.dtype)

        # Translate the function's body
        function_result: ValueExpr = self.visit(node.expr)[0]
        forwarding_tasklet = self.entry_state.add_tasklet(
            name="forwarding",
            inputs={f"{function_result.value.data}_internal"},
            outputs={f"{result}_internal" for result in result_names},
            code=f"{result_names[0]}_internal = {function_result.value.data}_internal",
            language=dace.dtypes.Language.Python,
        )
        function_result_memlet = create_memlet_full(
            function_result.value.data, self.sdfg.arrays[function_result.value.data]
        )
        self.entry_state.add_edge(
            function_result.value,
            None,
            forwarding_tasklet,
            f"{function_result.value.data}_internal",
            function_result_memlet,
        )

        output_accesses: list[dace.nodes.AccessNode] = [
            self.entry_state.add_access(name) for name in result_names
        ]
        for access in output_accesses:
            name = access.data
            ndim = len(self.sdfg.arrays[access.data].shape)
            memlet = create_memlet_at(name, tuple(["0"] * ndim))
            self.entry_state.add_edge(forwarding_tasklet, f"{name}_internal", access, None, memlet)

    def visit_SymRef(
        self, node: itir.SymRef, *, hack_is_iterator=False, **kwargs
    ) -> list[ValueExpr] | IteratorExpr:
        assert self.entry_state is not None
        access_node = self.entry_state.add_access(str(node.id))
        if hack_is_iterator:
            index = {
                dim: self.add_expr_tasklet([], idx, dace.dtypes.int64, idx)[0].value
                for dim, idx in self.domain.items()
            }
            return IteratorExpr(access_node, index)
        return [ValueExpr(access_node)]

    def visit_Literal(self, node: itir.Literal, **kwargs) -> list[ValueExpr]:
        value = node.value
        expr = str(value)
        dtype = _TYPE_MAPPING[node.type]
        return self.add_expr_tasklet([], expr, dtype, "constant")

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> Any:
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        elif (
            isinstance(node.fun, itir.FunCall)
            and isinstance(node.fun.fun, itir.SymRef)
            and node.fun.fun.id == "shift"
        ):
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
        elif isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in _MATH_BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            elif str(node.fun.id) in _GENERAL_BUILTIN_MAPPING:
                return self._visit_general_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def _visit_deref(self, node: itir.FunCall, **kwargs) -> list[ValueExpr]:
        iterator = self.visit(node.args[0], hack_is_iterator=True)
        flat_index = iterator.indices.items()
        flat_index = sorted(flat_index, key=lambda x: x[0])
        flat_index = [ValueExpr(x[1]) for x in flat_index]

        args: list[ValueExpr] = [ValueExpr(iterator.field), *flat_index]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[0]}[{', '.join(internals[1:])}]"
        return self.add_expr_tasklet(list(zip(args, internals)), expr, dace.dtypes.float64, "deref")

    def _split_shift_args(
        self, args: list[itir.Expr]
    ) -> tuple[list[itir.Expr], Optional[list[itir.Expr]]]:
        idx = 1
        for _ in range(idx, len(args)):
            if not isinstance(args[idx], itir.OffsetLiteral):
                idx += 1
        head = args[0:idx]
        rest = args[idx::] if idx < len(args) else None
        return head, rest

    def _make_shift_for_rest(self, rest, iterator):
        return itir.FunCall(
            fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=rest), args=[iterator]
        )

    def _visit_direct_addressing(self, node: itir.FunCall, **kwargs) -> IteratorExpr:
        assert isinstance(node.fun, itir.FunCall)
        shift = node.fun
        assert isinstance(shift, itir.FunCall)

        head, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0], hack_is_iterator=True)

        assert isinstance(head[0], itir.OffsetLiteral)
        offset = head[0].value
        assert isinstance(offset, str)
        shifted_dim = self.offset_provider[offset].value

        shift_amount = self.visit(head[1])

        args = [ValueExpr(iterator.indices[shifted_dim]), *shift_amount]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[0]} + {internals[1]}"
        shifted_value = self.add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "dir_addr"
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        shifted_index[shifted_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index)

    def _visit_indirect_addressing(self, node: itir.FunCall, **kwargs) -> IteratorExpr:
        shift = node.fun
        assert isinstance(shift, itir.FunCall)
        head, rest = self._split_shift_args(shift.args)
        if rest:
            iterator = self.visit(self._make_shift_for_rest(rest, node.args[0]))
        else:
            iterator = self.visit(node.args[0], hack_is_iterator=True)

        if len(head) < 2:
            raise NotImplementedError("reductions are not supported")
        assert isinstance(head[0], itir.OffsetLiteral)
        offset = head[0].value
        assert isinstance(offset, str)
        element = self.visit(head[1])

        table: NeighborTableOffsetProvider = self.offset_provider[offset]
        shifted_dim = table.origin_axis.value
        target_dim = table.neighbor_axis.value

        assert self.entry_state is not None
        conn = self.entry_state.add_access(connectivity_identifier(offset))

        args = [ValueExpr(conn), ValueExpr(iterator.indices[shifted_dim]), *element]
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = f"{internals[0]}[{internals[1]}, {internals[2]}]"
        shifted_value = self.add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "ind_addr"
        )[0].value

        shifted_index = {dim: value for dim, value in iterator.indices.items()}
        del shifted_index[shifted_dim]
        shifted_index[target_dim] = shifted_value

        return IteratorExpr(iterator.field, shifted_index)

    def _visit_numeric_builtin(self, node: itir.FunCall, **kwargs) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        fmt = _MATH_BUILTINS_MAPPING[str(node.fun.id)]
        args: list[ValueExpr] = list(itertools.chain(*[self.visit(arg) for arg in node.args]))
        internals = [f"{arg.value.data}_v" for arg in args]
        expr = fmt.format(*internals)
        return self.add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.float64, "numeric"
        )

    def _visit_general_builtin(self, node: itir.FunCall, **kwargs) -> list[ValueExpr]:
        assert isinstance(node.fun, itir.SymRef)
        expr_func = _GENERAL_BUILTIN_MAPPING[str(node.fun.id)]
        return expr_func(self, node.args)

    def add_expr_tasklet(
        self, args: list[tuple[ValueExpr, str]], expr: str, result_type: Any, name: str
    ) -> list[ValueExpr]:
        assert self.entry_state is not None
        assert self.sdfg is not None
        result_name = self.sdfg.temp_data_name() + "_tl"
        self.sdfg.add_scalar(result_name, result_type, transient=True)
        result_access = self.entry_state.add_access(result_name)

        expr_tasklet = self.entry_state.add_tasklet(
            name=name,
            inputs={internal for _, internal in args},
            outputs={"__result"},
            code=f"__result = {expr}",
        )

        for arg, internal in args:
            memlet = create_memlet_full(arg.value.data, self.sdfg.arrays[arg.value.data])
            self.entry_state.add_memlet_path(
                arg.value, expr_tasklet, memlet=memlet, src_conn=None, dst_conn=internal
            )

        self.entry_state.add_memlet_path(
            expr_tasklet,
            result_access,
            memlet=create_memlet_at(result_access.data, ("0",)),
            src_conn="__result",
            dst_conn=None,
        )

        return [ValueExpr(result_access)]


def closure_to_tasklet_sdfg(
    node: itir.StencilClosure,
    offset_provider: dict[str, Any],
    domain: dict[str, str],
    input_args: Sequence[ClosureArg],
    output_args: Sequence[ClosureArg],
    connectivity_args: Sequence[ClosureArg],
) -> tuple[dace.SDFG, list[str], list[str]]:
    translator = PythonTaskletCodegen(
        offset_provider, domain, input_args, output_args, connectivity_args
    )
    translator.visit(node.stencil)
    return translator.sdfg, translator.params, translator.results
