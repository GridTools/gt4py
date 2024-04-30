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
from typing import Any, final

import dace
import numpy as np

from gt4py.eve import codegen
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type
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


@dataclasses.dataclass
class SymbolExpr:
    value: dace.symbolic.SymbolicType
    dtype: dace.typeclass


@dataclasses.dataclass
class ValueExpr:
    value: dace.nodes.AccessNode
    dtype: dace.typeclass


class GtirTaskletCodegen(codegen.TemplatedGenerator):
    _sdfg: dace.SDFG
    _state: dace.SDFGState

    def __init__(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        self._sdfg = sdfg
        self._state = state

    @final
    def __call__(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """ "Creates the dataflow representing the given GTIR builtin.

        Returns a list of connections, where each connectio is defined as:
        tuple(node, connector_name)
        """
        return self._build()

    @final
    def _add_local_storage(self, data_type: ts.DataType, shape: list[str]) -> dace.nodes.AccessNode:
        name = f"{self._state.label}_var"
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = as_dace_type(data_type.dtype)
            name, _ = self._sdfg.add_array(name, shape, dtype, find_new_name=True, transient=True)
        else:
            assert isinstance(data_type, ts.ScalarType)
            assert len(shape) == 0
            dtype = as_dace_type(data_type)
            name, _ = self._sdfg.add_scalar(name, dtype, find_new_name=True, transient=True)
        return self._state.add_access(name)

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        raise NotImplementedError

    def _visit_deref(self, node: itir.FunCall) -> str:
        assert len(node.args) == 1
        if isinstance(node.args[0], itir.SymRef):
            return self.visit(node.args[0])
        raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")

    def _visit_numeric_builtin(self, node: itir.FunCall) -> str:
        assert isinstance(node.fun, itir.SymRef)
        fmt = _MATH_BUILTINS_MAPPING[str(node.fun.id)]
        args = self.visit(node.args)
        return fmt.format(*args)

    def visit_FunCall(self, node: itir.FunCall) -> str:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)
        elif isinstance(node.fun, itir.SymRef):
            builtin_name = str(node.fun.id)
            if builtin_name in _MATH_BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            else:
                raise NotImplementedError(f"'{builtin_name}' not implemented.")
        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

    @final
    def visit_Lambda(self, node: itir.Lambda) -> Any:
        # This visitor class should never encounter `itir.Lambda` expressions
        raise RuntimeError("Unexpected 'itir.Lambda' node encountered by 'GtirTaskletCodegen'.")

    @final
    def visit_Literal(self, node: itir.Literal) -> str:
        return node.value

    @final
    def visit_SymRef(self, node: itir.SymRef) -> str:
        return str(node.id)
