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
    field_type: ts.FieldType | ts.ScalarType


# Define alias for the elements needed to setup input connections to a map scope
InputConnection: TypeAlias = tuple[
    dace.nodes.AccessNode,
    sbs.Range,
    dace.nodes.Node,
    Optional[str],
]

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


class LambdaToTasklet(eve.NodeVisitor):
    """Translates an `ir.Lambda` expression to a dataflow graph.

    Lambda functions should only be encountered as argument to the `as_field_op`
    builtin function, therefore the dataflow graph generated here typically
    represents the stencil function of a field operator.
    """

    sdfg: dace.SDFG
    state: dace.SDFGState
    offset_provider: dict[str, Connectivity | Dimension]
    input_connections: list[InputConnection]
    symbol_map: dict[str, IteratorExpr | MemletExpr | SymbolExpr]

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        offset_provider: dict[str, Connectivity | Dimension],
    ):
        self.sdfg = sdfg
        self.state = state
        self.offset_provider = offset_provider
        self.input_connections = []
        self.symbol_map = {}

    def _add_entry_memlet_path(
        self,
        src: dace.nodes.AccessNode,
        src_subset: sbs.Range,
        dst_node: dace.nodes.Node,
        dst_conn: Optional[str] = None,
    ) -> None:
        self.input_connections.append((src, src_subset, dst_node, dst_conn))

    def _get_tasklet_result(
        self,
        dtype: dace.typeclass,
        src_node: dace.nodes.Tasklet,
        src_connector: str,
    ) -> ValueExpr:
        temp_name = self.sdfg.temp_data_name()
        self.sdfg.add_scalar(temp_name, dtype, transient=True)
        data_type = dace_fieldview_util.as_scalar_type(str(dtype.as_numpy_dtype()))
        temp_node = self.state.add_access(temp_name)
        self.state.add_edge(
            src_node,
            src_connector,
            temp_node,
            None,
            dace.Memlet(data=temp_name, subset="0"),
        )
        return ValueExpr(temp_node, data_type)

    def _visit_deref(self, node: itir.FunCall) -> MemletExpr | ValueExpr:
        assert len(node.args) == 1
        it = self.visit(node.args[0])

        if isinstance(it, IteratorExpr):
            field_desc = it.field.desc(self.sdfg)
            assert len(field_desc.shape) == len(it.dimensions)
            if all(isinstance(index, SymbolExpr) for index in it.indices.values()):
                # when all indices are symblic expressions, we can perform direct field access through a memlet
                field_subset = sbs.Indices([it.indices[dim].value for dim in it.dimensions])  # type: ignore[union-attr]
                return MemletExpr(it.field, field_subset)

            else:
                raise NotImplementedError

        else:
            assert isinstance(it, MemletExpr)
            return it

    def visit_FunCall(self, node: itir.FunCall) -> IteratorExpr | MemletExpr | ValueExpr:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)

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
                    arg_expr.subset,
                    tasklet_node,
                    connector,
                )

        # TODO: use type inference to determine the result type
        if len(node_connections) == 1:
            dtype = None
            for conn_name in ["__inp_0", "__inp_1"]:
                if conn_name in node_connections:
                    dtype = node_connections[conn_name].node.desc(self.sdfg).dtype
                    break
            if dtype is None:
                raise ValueError("Failed to determine the type")
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
            self._add_entry_memlet_path(
                output_expr.node,
                output_expr.subset,
                tasklet_node,
                "__inp",
            )
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
