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


from dataclasses import dataclass
from typing import Callable, TypeAlias

import dace
import numpy as np

from gt4py import eve
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtins.gtir_builtin_translator import (
    GTIRBuiltinTranslator,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import (
    as_dace_type,
    get_domain,
    unique_name,
)
from gt4py.next.type_system import type_specifications as ts


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


@dataclass(frozen=True)
class LiteralExpr:
    """Any symbolic expression that can be evaluated at compile time."""

    value: dace.symbolic.SymbolicType


@dataclass(frozen=True)
class SymbolExpr:
    """The data access to a scalar or field through a symbolic reference."""

    data: str


@dataclass(frozen=True)
class ValueExpr:
    """The result of a computation provided by a tasklet node."""

    node: dace.nodes.Tasklet
    connector: str


class GTIRBuiltinAsFieldOp(GTIRBuiltinTranslator, eve.NodeVisitor):
    """Generates the dataflow subgraph for the `as_field_op` builtin function."""

    TaskletConnector: TypeAlias = tuple[dace.nodes.Tasklet, str]

    stencil_expr: itir.Lambda
    stencil_args: list[Callable]
    field_domain: dict[Dimension, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]
    field_type: ts.FieldType
    input_connections: list[TaskletConnector]
    offset_provider: dict[str, Connectivity | Dimension]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        node: itir.FunCall,
        stencil_args: list[Callable],
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        super().__init__(sdfg, state)
        self.input_connections = []
        self.offset_provider = offset_provider

        assert cpm.is_call_to(node.fun, "as_fieldop")
        assert len(node.fun.args) == 2
        stencil_expr, domain_expr = node.fun.args
        # expect stencil (represented as a lambda function) as first argument
        assert isinstance(stencil_expr, itir.Lambda)
        # the domain of the field operator is passed as second argument
        assert isinstance(domain_expr, itir.FunCall)

        domain = get_domain(domain_expr)
        # define field domain with all dimensions in alphabetical order
        sorted_domain_dims = sorted(domain.keys(), key=lambda x: x.value)

        # add local storage to compute the field operator over the given domain
        # TODO: use type inference to determine the result type
        node_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)

        self.field_domain = domain
        self.field_type = ts.FieldType(sorted_domain_dims, node_type)
        self.stencil_expr = stencil_expr
        self.stencil_args = stencil_args

    def build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        assert len(self.input_connections) == 0

        # generate a tasklet node implementing the stencil function and represent
        # the field operator as a mapped tasklet, which will range over the field domain
        output_expr = self.visit(self.stencil_expr.expr)
        assert isinstance(output_expr, ValueExpr)

        # allocate local temporary storage for the result field
        field_shape = [
            # diff between upper and lower bound
            self.field_domain[dim][1] - self.field_domain[dim][0]
            for dim in self.field_type.dims
        ]
        field_node = self.add_local_storage(self.field_type, field_shape)

        data_nodes: dict[str, dace.nodes.AccessNode] = {}
        input_memlets: dict[str, dace.Memlet] = {}
        for arg, param in zip(self.stencil_args, self.stencil_expr.params, strict=True):
            arg_nodes = arg()
            assert len(arg_nodes) == 1
            arg_node, arg_type = arg_nodes[0]
            data = str(param.id)
            # require (for now) all input nodes to be data access nodes
            assert isinstance(arg_node, dace.nodes.AccessNode)
            data_nodes[data] = arg_node
            if isinstance(arg_type, ts.FieldType):
                # support either single element access (general case) or full array shape
                is_scalar = all(dim in self.field_domain for dim in arg_type.dims)
                if is_scalar:
                    subset = ",".join(f"i_{dim.value}" for dim in arg_type.dims)
                    input_memlets[data] = dace.Memlet(data=arg_node.data, subset=subset, volume=1)
                else:
                    memlet = dace.Memlet.from_array(arg_node.data, arg_node.desc(self.sdfg))
                    # set volume to 1 because the stencil function always performs single element access
                    # TODO: check validity of this assumption
                    memlet.volume = 1
                    input_memlets[data] = memlet
            else:
                input_memlets[data] = dace.Memlet(data=arg_node.data, subset="0")

        # assume tasklet with single output
        output_index = ",".join(f"i_{dim.value}" for dim in self.field_type.dims)
        output_memlet = dace.Memlet(data=field_node.data, subset=output_index)

        # create map range corresponding to the field operator domain
        map_ranges = {f"i_{dim.value}": f"{lb}:{ub}" for dim, (lb, ub) in self.field_domain.items()}
        me, mx = self.head_state.add_map(unique_name("map"), map_ranges)

        for (input_node, input_connector), input_param in self.input_connections:
            assert input_param in data_nodes
            self.head_state.add_memlet_path(
                data_nodes[input_param],
                me,
                input_node,
                dst_conn=input_connector,
                memlet=input_memlets[input_param],
            )
        self.head_state.add_memlet_path(
            output_expr.node, mx, field_node, src_conn=output_expr.connector, memlet=output_memlet
        )

        return [(field_node, self.field_type)]

    def _visit_shift(self, node: itir.FunCall) -> str:
        raise NotImplementedError

    def visit_FunCall(self, node: itir.FunCall) -> ValueExpr | SymbolExpr:
        inp_tasklets = {}
        inp_symbols = set()
        inp_connectors = set()
        node_internals = []
        for i, arg in enumerate(node.args):
            arg_expr = self.visit(arg)
            if isinstance(arg_expr, LiteralExpr):
                # use the value without adding any connector
                node_internals.append(arg_expr.value)
            else:
                if isinstance(arg_expr, ValueExpr):
                    # the value is the result of a tasklet node
                    connector = f"__inp_{i}"
                    inp_tasklets[connector] = arg_expr
                else:
                    # the value is the result of a tasklet node
                    assert isinstance(arg_expr, SymbolExpr)
                    connector = f"__inp_{arg_expr.data}"
                    inp_symbols.add((connector, arg_expr.data))
                inp_connectors.add(connector)
                node_internals.append(connector)

        if cpm.is_call_to(node, "deref"):
            assert len(inp_tasklets) == 0
            assert len(inp_symbols) == 1
            _, data = inp_symbols.pop()
            return SymbolExpr(data)

        elif cpm.is_call_to(node.fun, "shift"):
            code = self._visit_shift(node.fun)

        elif isinstance(node.fun, itir.SymRef):
            # create a tasklet node implementing the builtin function
            builtin_name = str(node.fun.id)
            if builtin_name in _MATH_BUILTINS_MAPPING:
                fmt = _MATH_BUILTINS_MAPPING[builtin_name]
                code = fmt.format(*node_internals)
            else:
                raise NotImplementedError(f"'{builtin_name}' not implemented.")

            out_connector = "__out"
            tasklet_node = self.head_state.add_tasklet(
                unique_name("tasklet"),
                inp_connectors,
                {out_connector},
                "{} = {}".format(out_connector, code),
            )

        else:
            raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

        for input_conn, inp_expr in inp_tasklets.items():
            self.head_state.add_edge(
                inp_expr.node, inp_expr.connector, tasklet_node, input_conn, dace.Memlet()
            )
        self.input_connections.extend(
            ((tasklet_node, connector), data) for connector, data in inp_symbols
        )
        return ValueExpr(tasklet_node, out_connector)

    def visit_Literal(self, node: itir.Literal) -> LiteralExpr:
        cast_sym = str(as_dace_type(node.type))
        cast_fmt = _MATH_BUILTINS_MAPPING[cast_sym]
        typed_value = cast_fmt.format(node.value)
        return LiteralExpr(typed_value)

    def visit_SymRef(self, node: itir.SymRef) -> SymbolExpr:
        """
        Symbol references are mapped to tasklet connectors that access some kind of data.
        """
        sym_name = str(node.id)
        return SymbolExpr(sym_name)
