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


from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias, final

import dace
import numpy as np

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name
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
    value: dace.symbolic.SymbolicType


@dataclass(frozen=True)
class SymbolExpr:
    data: str


@dataclass(frozen=True)
class ValueExpr:
    node: dace.nodes.Tasklet
    connector: str


class GTIRBuiltinTranslator(eve.NodeVisitor):
    TaskletConnector: TypeAlias = tuple[dace.nodes.Tasklet, str]

    sdfg: dace.SDFG
    head_state: dace.SDFGState
    input_connections: list[TaskletConnector]

    def __init__(self, sdfg: dace.SDFG, head_state: dace.SDFGState):
        self.sdfg = sdfg
        self.head_state = head_state
        self.input_connections = []

    @final
    def __call__(
        self,
    ) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """The callable interface is used to build the dataflow graph.

        It allows to build the dataflow graph inside a given state starting
        from the innermost nodes, by propagating the intermediate results
        as access nodes to temporary local storage.
        """
        return self.build()

    @final
    def add_local_storage(
        self, data_type: ts.FieldType | ts.ScalarType, shape: list[str]
    ) -> dace.nodes.AccessNode:
        """Allocates temporary storage to be used in the local scope for intermediate results."""
        name = unique_name("var")
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = as_dace_type(data_type.dtype)
            name, _ = self.sdfg.add_array(name, shape, dtype, find_new_name=True, transient=True)
        else:
            assert len(shape) == 0
            dtype = as_dace_type(data_type)
            name, _ = self.sdfg.add_scalar(name, dtype, find_new_name=True, transient=True)
        return self.head_state.add_access(name)

    @abstractmethod
    def build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """Creates the dataflow subgraph representing a given GTIR builtin.

        This method is used by derived classes of `GTIRBuiltinTranslator`,
        which build a specialized subgraph for a certain GTIR builtin.

        Returns a list of SDFG nodes and the associated GT4Py data type:
        tuple(node, data_type)

        The GT4Py data type is useful in the case of fields, because it provides
        information on the field domain (e.g. order of dimensions, types of dimensions).
        """

    def _visit_deref(self, node: itir.FunCall) -> ValueExpr | SymbolExpr:
        assert len(node.args) == 1
        if isinstance(node.args[0], itir.SymRef):
            return self.visit(node.args[0])
        raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")

    @final
    def visit_FunCall(self, node: itir.FunCall) -> ValueExpr | SymbolExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

        elif isinstance(node.fun, itir.SymRef):
            # create a tasklet node implementing the builtin function
            inputs = {}
            inp_data = set()
            inp_nodes = set()
            node_internals = []
            for i, arg in enumerate(node.args):
                arg_expr = self.visit(arg)
                if isinstance(arg_expr, LiteralExpr):
                    node_internals.append(arg_expr.value)
                else:
                    connector = f"__inp_{i}"
                    if isinstance(arg_expr, ValueExpr):
                        inputs[connector] = arg_expr
                    else:
                        assert isinstance(arg_expr, SymbolExpr)
                        inp_data.add((connector, arg_expr.data))
                    inp_nodes.add(connector)
                    node_internals.append(connector)

            builtin_name = str(node.fun.id)
            if builtin_name in _MATH_BUILTINS_MAPPING:
                fmt = _MATH_BUILTINS_MAPPING[builtin_name]
                code = fmt.format(*node_internals)
            else:
                raise NotImplementedError(f"'{builtin_name}' not implemented.")

            out_connector = "__out"
            tasklet_node = self.head_state.add_tasklet(
                unique_name("tasklet"),
                inp_nodes,
                {out_connector},
                "{} = {}".format(out_connector, code),
            )
            for input_conn, inp_expr in inputs.items():
                self.head_state.add_edge(
                    inp_expr.node, inp_expr.connector, tasklet_node, input_conn, dace.Memlet()
                )
            self.input_connections.extend(
                ((tasklet_node, connector), data) for connector, data in inp_data
            )
            return ValueExpr(tasklet_node, out_connector)

        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

    @final
    def visit_Lambda(self, node: itir.Lambda) -> Any:
        """
        This visitor class should never encounter `itir.Lambda` expressions
        because a lambda represents a stencil, which operates from iterator to values.
        In fieldview, lambdas should only be arguments to field operators (`as_field_op`).
        """
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GTIRBuiltinTranslator'.")

    @final
    def visit_Literal(self, node: itir.Literal) -> LiteralExpr:
        cast_sym = str(as_dace_type(node.type))
        cast_fmt = _MATH_BUILTINS_MAPPING[cast_sym]
        typed_value = cast_fmt.format(node.value)
        return LiteralExpr(typed_value)

    @final
    def visit_SymRef(self, node: itir.SymRef) -> SymbolExpr:
        """
        Symbol references are mapped to tasklet connectors that access some kind of data.
        """
        sym_name = str(node.id)
        return SymbolExpr(sym_name)
