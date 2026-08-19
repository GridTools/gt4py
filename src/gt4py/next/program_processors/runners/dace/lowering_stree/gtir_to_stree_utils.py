# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for the GTIR-to-ScheduleTree lowering.

This module is self-contained: it does not import anything from the
``lowering`` package (the SDFG-based lowering).  All needed helpers —
domain parsing, symbolic utilities, Python code generation, and the
``_CONST_DIM`` constant — are duplicated here so that ``lowering_stree``
can evolve independently.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Final, Mapping, Optional, Sequence, TypeAlias, TypeVar

import dace
import numpy as np
import sympy
from dace import subsets as dace_subsets

from gt4py import eve
from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt
from gt4py.eve.extended_typing import MaybeNestedInTuple, NestedTuple
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import builtins, ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.type_system import type_specifications as ts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Magic local dimension used for list of values with length known at compile-time.
_CONST_DIM: Final = gtx_common.Dimension(
    value="_CONST_DIM", kind=gtx_common.DimensionKind.LOCAL
)

# Prefix string to be used for tasklet connectors.
_TASKLET_CONNECTOR_PREFIX: Final[str] = "__tlet_"


# ---------------------------------------------------------------------------
# Debug info
# ---------------------------------------------------------------------------


def debug_info(
    node: gtir.Node, *, default: Optional[dace.dtypes.DebugInfo] = None
) -> Optional[dace.dtypes.DebugInfo]:
    """Include the GT4Py node location as debug information."""
    location = node.location
    if location:
        return dace.dtypes.DebugInfo(
            start_line=location.line,
            start_column=location.column if location.column else 0,
            end_line=location.end_line if location.end_line else -1,
            end_column=location.end_column if location.end_column else 0,
            filename=location.filename,
        )
    return default


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def get_map_variable(dim: gtx_common.Dimension) -> str:
    """Format map variable name based on the naming convention for SDFG transformations."""
    dim = gtx_common.as_non_staggered(dim)
    suffix = "dim" if dim.kind == gtx_common.DimensionKind.LOCAL else ""
    return f"i_{dim.value}_gtx_{dim.kind}{suffix}"


def make_tasklet_connector_for(name: str) -> str:
    """Format tasklet connector name to avoid conflicts with GTIR program symbols."""
    assert _TASKLET_CONNECTOR_PREFIX.startswith("__")
    assert not name.startswith(_TASKLET_CONNECTOR_PREFIX)
    return f"{_TASKLET_CONNECTOR_PREFIX}{name.lstrip('_')}"


# ---------------------------------------------------------------------------
# Tuple helpers
# ---------------------------------------------------------------------------


def make_symbol_tree(tuple_name: str, tuple_type: ts.TupleType) -> NestedTuple[gtir.Sym]:
    """Creates a tree representation of the symbols corresponding to the tuple fields."""
    assert all(isinstance(t, ts.DataType) for t in tuple_type.types)
    fields = [(f"{tuple_name}_{i}", field_type) for i, field_type in enumerate(tuple_type.types)]
    return tuple(
        make_symbol_tree(field_name, field_type)
        if isinstance(field_type, ts.TupleType)
        else im.sym(field_name, field_type)
        for field_name, field_type in fields
    )


def flatten_tuple_fields(tuple_name: str, tuple_type: ts.TupleType) -> list[gtir.Sym]:
    """Creates a list of symbols, annotated with the data type, for all elements of the given tuple."""
    symbol_tree = make_symbol_tree(tuple_name, tuple_type)
    return list(gtx_utils.flatten_nested_tuple(symbol_tree))


# ---------------------------------------------------------------------------
# Symbol replacement
# ---------------------------------------------------------------------------


def replace_invalid_symbols(ir: gtir.Program) -> gtir.Program:
    """Ensure that all symbols used in the program IR are valid strings."""

    class ReplaceSymbols(eve.PreserveLocationVisitor, eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("domain",)

        T = TypeVar("T", gtir.Sym, gtir.SymRef)

        def _replace_sym(self, node: T, symtable: Dict[str, str]) -> T:
            sym = str(node.id)
            return type(node)(id=symtable.get(sym, sym), type=node.type)

        def visit_Sym(self, node: gtir.Sym, *, symtable: Dict[str, str]) -> gtir.Sym:
            return self._replace_sym(node, symtable)

        def visit_SymRef(self, node: gtir.SymRef, *, symtable: Dict[str, str]) -> gtir.SymRef:
            return self._replace_sym(node, symtable)

    if not all(dace.dtypes.validate_name(str(sym.id)) for sym in ir.params):
        raise ValueError("Invalid symbol in program parameters.")

    if any(str(sym.id).startswith(_TASKLET_CONNECTOR_PREFIX) for sym in ir.params):
        raise ValueError(
            f"Unexpectd symbol with prefix '{_TASKLET_CONNECTOR_PREFIX}' in program parameters."
        )

    ir_sym_ids = {str(sym.id) for sym in eve.walk_values(ir).if_isinstance(gtir.Sym).to_set()}
    ir_ssa_uuid = eve.utils.SequentialIDGenerator(prefix="gtir_var")

    invalid_symbols_mapping = {
        sym_id: next(ir_ssa_uuid)
        for sym_id in sorted(ir_sym_ids)
        if not dace.dtypes.validate_name(sym_id)
    }
    if len(invalid_symbols_mapping) == 0:
        return ir

    assert ir_sym_ids.isdisjoint(invalid_symbols_mapping.values())
    return ReplaceSymbols().visit(ir, symtable=invalid_symbols_mapping)


# ---------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------


def get_symbolic(ir: gtir.Expr) -> dace.symbolic.SymbolicType:
    """Convert a GTIR expression to a dace symbolic expression."""
    python_source = get_source(ir)
    return dace.symbolic.pystr_to_symbolic(python_source)


def safe_replace_symbolic(
    val: dace.symbolic.SymbolicType,
    symbol_mapping: Mapping[dace.symbolic.SymbolicType | str, dace.symbolic.SymbolicType | str],
) -> dace.symbolic.SymbolicType:
    """Replace free symbols in a dace symbolic expression, using ``safe_replace()``."""
    x = [val]
    dace.symbolic.safe_replace(symbol_mapping, lambda m, xx=x: xx.append(xx[-1].subs(m)))
    return x[-1]


# ---------------------------------------------------------------------------
# Domain helpers (duplicated from gtir_domain.py)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FieldopDomainRange:
    """Represents the range of a field operator domain in one dimension.

    Attributes:
        dim: dimension definition
        start: symbolic expression for lower bound (inclusive)
        stop: symbolic expression for upper bound (exclusive)
    """

    dim: gtx_common.Dimension
    start: dace.symbolic.SymbolicType
    stop: dace.symbolic.SymbolicType


FieldopDomain: TypeAlias = list[FieldopDomainRange]
"""Domain of a field operator represented as a list of FieldopDomainRange."""


TargetDomain: TypeAlias = MaybeNestedInTuple[domain_utils.SymbolicDomain]
"""Symbolic domain which defines the range to write in the target field."""


def get_field_domain(domain: domain_utils.SymbolicDomain) -> FieldopDomain:
    """Visit the domain of a field operator and return a list of dimensions and bounds.

    The returned lower bound is inclusive, the upper bound is exclusive.
    Domain dimensions are sorted in gt4py canonical order.
    """
    return [
        FieldopDomainRange(
            dim,
            get_symbolic(domain.ranges[dim].start),
            get_symbolic(domain.ranges[dim].stop),
        )
        for dim in gtx_common.order_dimensions(domain.ranges.keys())
    ]


def extract_target_domain(node: gtir.Expr) -> TargetDomain:
    """Visit a GTIR domain expression and construct a TargetDomain symbolic domain."""

    class TargetDomainParser(eve.visitors.NodeTranslator):
        def visit_FunCall(self, node: gtir.FunCall) -> TargetDomain:
            if cpm.is_call_to(node, "make_tuple"):
                return tuple(self.visit(arg) for arg in node.args)
            else:
                return domain_utils.SymbolicDomain.from_expr(node)

    return TargetDomainParser().visit(node)


def get_element_subset(
    dims: Sequence[gtx_common.Dimension], origin: Optional[Sequence[dace.symbolic.SymExpr]]
) -> dace_subsets.Range:
    """Construct the Memlet subset to access an element in the field domain."""
    assert len(dims) != 0
    index_variables = [dace.symbolic.pystr_to_symbolic(get_map_variable(dim)) for dim in dims]
    origin = [0] * len(index_variables) if origin is None else origin
    start_values = [
        index - start_index for index, start_index in zip(index_variables, origin, strict=True)
    ]
    return dace_subsets.Range([(s, s, 1) for s in start_values])


def get_field_layout(
    field_domain: FieldopDomain,
) -> tuple[list[gtx_common.Dimension], list[dace.symbolic.SymExpr], list[dace.symbolic.SymExpr]]:
    """Parse the field operator domain and generate the shape of the result field.

    Ensures that the array shape computed from the domain range is non-negative.

    Returns:
        A tuple of three lists: domain dimensions, domain origin, domain size.
    """
    if len(field_domain) == 0:
        return [], [], []
    domain_dims = [domain_range.dim for domain_range in field_domain]
    domain_origin = [domain_range.start for domain_range in field_domain]
    domain_shape = [
        dace.symbolic.pystr_to_symbolic(f"max(0, {domain_range.stop - domain_range.start})")
        for domain_range in field_domain
    ]
    return domain_dims, domain_origin, domain_shape


# ---------------------------------------------------------------------------
# Python code generation (duplicated from gtir_python_codegen.py)
# ---------------------------------------------------------------------------

MATH_BUILTINS_MAPPING: dict[str, str] = {
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


def _builtin_get_domain_range(field: str, axis: str) -> str:
    return f"__{field}_{axis}_range"


def _builtin_if(cond: str, true_val: str, false_val: str) -> str:
    return f"({true_val} if {cond} else {false_val})"


def _builtin_tuple_get(index: str, tuple_name: str) -> str:
    return f"{tuple_name}_{index}"


def _builtin_make_const_list(arg: str) -> str:
    return arg


GENERAL_BUILTIN_MAPPING: dict[str, Callable[..., str]] = {
    "cast_": _builtin_cast,
    "get_domain_range": _builtin_get_domain_range,
    "if_": _builtin_if,
    "make_const_list": _builtin_make_const_list,
    "tuple_get": _builtin_tuple_get,
}


def format_builtin(builtin: str, *args: Any) -> str:
    """Format a GTIR builtin as a Python code string."""
    if builtin in MATH_BUILTINS_MAPPING:
        fmt = MATH_BUILTINS_MAPPING[builtin]
        return fmt.format(*args)
    elif builtin in GENERAL_BUILTIN_MAPPING:
        expr_func = GENERAL_BUILTIN_MAPPING[builtin]
        return expr_func(*args)
    else:
        raise NotImplementedError(f"'{builtin}' not implemented.")


class PythonCodegen(codegen.TemplatedGenerator):
    """Helper class to visit a symbolic expression and translate it to Python code."""

    Literal = as_fmt("{value}")

    def visit_AxisLiteral(self, node: gtir.AxisLiteral, **kwargs: Any) -> str:
        return node.value

    def visit_FunCall(self, node: gtir.FunCall, args_map: dict[str, gtir.Node]) -> str:
        if isinstance(node.fun, gtir.Lambda):
            lambda_args_map = args_map | {
                p.id: arg for p, arg in zip(node.fun.params, node.args, strict=True)
            }
            return self.visit(node.fun.expr, args_map=lambda_args_map)
        elif cpm.is_call_to(node, "deref"):
            assert len(node.args) == 1
            if not isinstance(node.args[0], gtir.SymRef):
                raise NotImplementedError(f"Unexpected deref with arg type '{type(node.args[0])}'.")
            return self.visit(node.args[0], args_map=args_map)
        elif isinstance(node.fun, gtir.SymRef):
            args = self.visit(node.args, args_map=args_map)
            builtin_name = str(node.fun.id)
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
    """Visit a symbolic expression and return the corresponding Python code string."""
    return PythonCodegen.apply(node, args_map={})
