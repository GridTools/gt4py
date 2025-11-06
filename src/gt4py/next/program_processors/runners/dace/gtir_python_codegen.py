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
import sympy

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt
from gt4py.next import common as gtx_common
from gt4py.next.iterator import builtins, ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils


MATH_BUILTINS_MAPPING = {
    "abs": "abs({})",
    "neg": "(- {})",
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
    "and_": "({} and {})",
    "or_": "({} or {})",
    "xor_": "({} != {})",
    "mod": "({} % {})",
    "not_": "(not {})",
}


def _builtin_cast(val: str, target_type: str) -> str:
    assert target_type in builtins.TYPE_BUILTINS
    return MATH_BUILTINS_MAPPING[target_type].format(val)


def _builtin_if(cond: str, true_val: str, false_val: str) -> str:
    return f"{true_val} if {cond} else {false_val}"


def _builtin_tuple_get(index: str, tuple_name: str) -> str:
    return f"{tuple_name}_{index}"


def _make_const_list(arg: str) -> str:
    """
    Takes a single scalar argument and broadcasts this value on the local dimension
    of map expression. In a dataflow, we represent it as a tasklet that writes
    a value to a scalar node.
    """
    return arg


GENERAL_BUILTIN_MAPPING: dict[str, Callable[..., str]] = {
    "cast_": _builtin_cast,
    "if_": _builtin_if,
    "make_const_list": _make_const_list,
    "tuple_get": _builtin_tuple_get,
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

    Literal = as_fmt("{value}")

    def _visit_access_to_domain_range(
        self, node: gtir.FunCall, args_map: dict[str, gtir.Node]
    ) -> str:
        assert cpm.is_call_to(node, "tuple_get")
        index, tuple_ = node.args
        assert isinstance(index, gtir.Literal)
        assert cpm.is_call_to(tuple_, "get_domain_range")
        field, axis = tuple_.args
        assert isinstance(axis, gtir.AxisLiteral)
        dim = gtx_common.Dimension(axis.value)
        field_name = self.visit(field, args_map=args_map)
        # The 'get_domain_range' builtin function returns a tuple of two values, the
        # range start and stop. Combined with 'tuple_get', this function will build
        # the symbolic expression to retrieve of the two values.
        origin = gtx_dace_utils.field_origin_symbol(field_name, dim)
        size = gtx_dace_utils.field_size_symbol(field_name, dim)
        if index.value == "0":
            return origin.name
        elif index.value == "1":
            return f"{origin.name} + {size.name}"
        else:
            raise ValueError(f"Unxpect 'tuple_get' on domain range with index '{index.value}'.")

    def visit_FunCall(self, node: gtir.FunCall, args_map: dict[str, gtir.Node]) -> str:
        if isinstance(node.fun, gtir.Lambda):
            # update the mapping from lambda parameters to corresponding argument expressions
            lambda_args_map = args_map | {
                p.id: arg for p, arg in zip(node.fun.params, node.args, strict=True)
            }
            return self.visit(node.fun.expr, args_map=lambda_args_map)
        elif cpm.is_call_to(node, "deref"):
            assert len(node.args) == 1
            if not isinstance(node.args[0], gtir.SymRef):
                # shift expressions are not expected in this visitor context
                raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")
            return self.visit(node.args[0], args_map=args_map)
        elif cpm.is_call_to(node, "tuple_get") and cpm.is_call_to(node.args[1], "get_domain_range"):
            # special handling is needed to retrieve the SDFG symbols corresponding to field range

            return self._visit_access_to_domain_range(node, args_map=args_map)
        elif isinstance(node.fun, gtir.SymRef):
            builtin_name = str(node.fun.id)
            args = self.visit(node.args, args_map=args_map)
            return format_builtin(builtin_name, *args)
        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

    def visit_InfinityLiteral(self, node: gtir.InfinityLiteral, **kwargs: Any) -> str:
        return str(sympy.oo) if node == gtir.InfinityLiteral.POSITIVE else str(-sympy.oo)

    def visit_SymRef(self, node: gtir.SymRef, args_map: dict[str, gtir.Node]) -> str:
        symbol = str(node.id)
        if symbol in args_map:
            return self.visit(args_map[symbol], args_map=args_map)
        return symbol


def get_source(node: gtir.Node) -> str:
    """
    Specialized visit method for symbolic expressions.

    The visitor uses `args_map` to map lambda parameters to the corresponding argument expressions.

    Returns:
        A string containing the Python code corresponding to a symbolic expression
    """
    return PythonCodegen.apply(node, args_map={})
