# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm


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


def builtin_cast(*args: Any) -> str:
    val, target_type = args
    return MATH_BUILTINS_MAPPING[target_type].format(val)


def builtin_if(*args: Any) -> str:
    cond, true_val, false_val = args
    return f"{true_val} if {cond} else {false_val}"


def make_const_list(arg: str) -> str:
    return arg


GENERAL_BUILTIN_MAPPING: dict[str, Callable[[Any], str]] = {
    "cast_": builtin_cast,
    "if_": builtin_if,
    "make_const_list": make_const_list,
}


def format_builtin(builtin: str, *args: Any) -> str:
    if builtin in MATH_BUILTINS_MAPPING:
        fmt = MATH_BUILTINS_MAPPING[builtin]
        return fmt.format(*args)
    elif builtin in GENERAL_BUILTIN_MAPPING:
        expr_func = GENERAL_BUILTIN_MAPPING[builtin]
        return expr_func(*args)
    else:
        raise NotImplementedError(f"'{builtin}' not implemented.")


class PythonCodegen(codegen.TemplatedGenerator):
    """Helper class to visit a symbolic expression and translate it to Python code.

    The generated Python code can be use either as the body of a tasklet node or,
    as in the case of field domain definitions, for sybolic array shape and map range.
    """

    SymRef = as_fmt("{id}")
    Literal = as_fmt("{value}")

    def _visit_deref(self, node: gtir.FunCall) -> str:
        assert len(node.args) == 1
        if isinstance(node.args[0], gtir.SymRef):
            return self.visit(node.args[0])
        raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")

    def visit_FunCall(self, node: gtir.FunCall) -> str:
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node)
        elif isinstance(node.fun, gtir.SymRef):
            args = self.visit(node.args)
            builtin_name = str(node.fun.id)
            return format_builtin(builtin_name, *args)
        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")


get_source = PythonCodegen.apply
"""
Specialized visit method for symbolic expressions.

Returns:
    A string containing the Python code corresponding to a symbolic expression
"""
