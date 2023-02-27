# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from collections.abc import Sequence
from typing import Any, Optional

import dace
import numpy as np

import gt4py.eve.codegen
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from .utility import connectivity_identifier, create_memlet_at, create_memlet_full


_BUILTINS_MAPPING = {
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


class PythonTaskletCodegen(gt4py.eve.codegen.TemplatedGenerator):
    sdfg: Optional[dace.SDFG]
    entry_state: Optional[dace.SDFGState]
    offset_provider: dict[str, Any]
    domain: dict[str, str]
    input_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]]
    output_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]]
    conn_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]]
    params: list[str]
    results: list[str]

    def __init__(
        self,
        offset_provider: dict[str, Any],
        domain: dict[str, str],
        input_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
        output_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
        conn_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
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
        conn_names = [conn[0].data for conn in self.conn_args]

        self.params = param_names
        self.results = result_names

        # Create the SDFG for the function's body
        self.sdfg = dace.SDFG(func_name)
        self.entry_state = self.sdfg.add_state(f"{func_name}_entry", True)

        # Add input and output parameters as arrays
        array_names = [*param_names, *result_names, *conn_names]
        args = [*self.input_args, *self.output_args, *self.conn_args]
        for name, (memlet, dtype) in zip(array_names, args):
            shape = memlet.subset.size()
            self.sdfg.add_array(name, shape=shape, dtype=dtype)

        # Translate the function's body
        function_result: dace.nodes.AccessNode = self.visit(node.expr)
        forwarding_tasklet = self.entry_state.add_tasklet(
            name="forwarding",
            inputs={f"{function_result.data}_internal"},
            outputs={f"{result}_internal" for result in result_names},
            code=f"{result_names[0]}_internal = {function_result.data}_internal",
            language=dace.dtypes.Language.Python,
        )
        function_result_memlet = create_memlet_full(function_result.data, self.sdfg.arrays[function_result.data])
        self.entry_state.add_edge(function_result, None, forwarding_tasklet, f"{function_result.data}_internal", function_result_memlet)

        output_accesses: list[dace.nodes.AccessNode] = [self.entry_state.add_access(name) for name in result_names]
        for access in output_accesses:
            name = access.data
            ndim = len(self.sdfg.arrays[access.data].shape)
            memlet = create_memlet_at(name, tuple(["0"] * ndim))
            self.entry_state.add_edge(forwarding_tasklet, f"{name}_internal", access, None, memlet)

    def visit_SymRef(
        self, node: itir.SymRef, *, hack_is_iterator=False, **kwargs
    ) -> dace.nodes.AccessNode | tuple[dace.nodes.AccessNode, dict[str, dace.nodes.AccessNode]]:
        assert self.entry_state is not None
        access_node = self.entry_state.add_access(str(node.id))
        if hack_is_iterator:
            index = {
                dim: self._add_expr_tasklet([], idx, dace.dtypes.int64, idx)
                for dim, idx in self.domain.items()
            }
            return access_node, index
        return access_node

    def visit_Literal(self, node: itir.Literal, **kwargs) -> dace.nodes.AccessNode:
        value = node.value
        expr = str(value)
        dtype = dace.dtypes.float64 if np.dtype(type(value)).kind == "f" else dace.dtypes.int64
        return self._add_expr_tasklet([], expr, dtype, "constant")

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
            if str(node.fun.id) in _BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def _visit_deref(self, node: itir.FunCall, **kwargs) -> dace.nodes.AccessNode:
        iterator = node.args[0]
        sym, index = self.visit(iterator, hack_is_iterator=True)
        flat_index = index.items()
        flat_index = sorted(flat_index, key=lambda x: x[0])
        flat_index = [x[1] for x in flat_index]

        args: list[dace.nodes.AccessNode] = [sym, *flat_index]
        internals = [f"{arg.data}_v" for arg in args]
        expr = f"{internals[0]}[{', '.join(internals[1:])}]"
        return self._add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.float64, "deref"
        )

    def _visit_direct_addressing(
        self, node: itir.FunCall, **kwargs
    ) -> tuple[dace.nodes.AccessNode, dict[str, dace.nodes.AccessNode]]:
        assert isinstance(node.fun, itir.FunCall)
        iterator = node.args[0]
        sym, index = self.visit(iterator, hack_is_iterator=True)
        assert isinstance(node.fun.args[0], itir.OffsetLiteral)
        offset = node.fun.args[0].value
        assert isinstance(offset, str)
        shifted_dim = self.offset_provider[offset].value

        shift_amount = self.visit(node.fun.args[1])

        args = [index[shifted_dim], shift_amount]
        internals = [f"{arg.data}_v" for arg in args]
        expr = f"{internals[0]} + {internals[1]}"
        shifted_value = self._add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "dir_addr"
        )

        shifted_index = {dim: value for dim, value in index.items()}
        shifted_index[shifted_dim] = shifted_value

        return sym, shifted_index

    def _visit_indirect_addressing(
        self, node: itir.FunCall, **kwargs
    ) -> tuple[dace.nodes.AccessNode, dict[str, str]]:
        iterator = node.args[0]
        sym, index = self.visit(iterator, hack_is_iterator=True)

        assert isinstance(node.fun, itir.FunCall)
        assert isinstance(node.fun.args[0], itir.OffsetLiteral)
        offset = node.fun.args[0].value
        assert isinstance(offset, str)
        element = self.visit(node.fun.args[1])

        table: NeighborTableOffsetProvider = self.offset_provider[offset]
        shifted_dim = table.origin_axis.value
        target_dim = table.neighbor_axis.value

        assert self.entry_state is not None
        conn = self.entry_state.add_access(connectivity_identifier(offset))

        args = [conn, index[shifted_dim], element]
        internals = [f"{arg.data}_v" for arg in args]
        expr = f"{internals[0]}[{internals[1]}, {internals[2]}]"
        shifted_value = self._add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.int64, "ind_addr"
        )

        shifted_index = {dim: value for dim, value in index.items()}
        del shifted_index[shifted_dim]
        shifted_index[target_dim] = shifted_value

        return sym, shifted_index

    def _visit_numeric_builtin(self, node: itir.FunCall, **kwargs) -> dace.nodes.AccessNode:
        assert isinstance(node.fun, itir.SymRef)
        fmt = _BUILTINS_MAPPING[str(node.fun.id)]
        args: list[dace.nodes.AccessNode] = self.visit(node.args)
        internals = [f"{arg.data}_v" for arg in args]
        expr = fmt.format(*internals)
        return self._add_expr_tasklet(
            list(zip(args, internals)), expr, dace.dtypes.float64, "numeric"
        )

    def _add_expr_tasklet(
        self, args: list[tuple[dace.nodes.AccessNode, str]], expr: str, result_type: Any, name: str
    ) -> dace.nodes.AccessNode:
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

        for access, internal in args:
            memlet = create_memlet_full(access.data, self.sdfg.arrays[access.data])
            self.entry_state.add_memlet_path(
                access, expr_tasklet, memlet=memlet, src_conn=None, dst_conn=internal
            )

        self.entry_state.add_memlet_path(
            expr_tasklet,
            result_access,
            memlet=create_memlet_at(result_access.data, ("0",)),
            src_conn="__result",
            dst_conn=None,
        )

        return result_access


def closure_to_tasklet_sdfg(
    node: itir.StencilClosure,
    offset_provider: dict[str, Any],
    domain: dict[str, str],
    input_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
    output_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
    connectivity_args: Sequence[tuple[dace.Memlet, dace.dtypes.typeclass]],
) -> tuple[dace.SDFG, list[str], list[str]]:
    translator = PythonTaskletCodegen(
        offset_provider, domain, input_args, output_args, connectivity_args
    )
    translator.visit(node.stencil)
    return translator.sdfg, translator.params, translator.results
