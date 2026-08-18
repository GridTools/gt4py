# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Python code generator for the schedule-tree lowering.

This module is the schedule-tree analogue of ``lowering.gtir_python_codegen``
combined with ``lowering.gtir_to_sdfg_iterator``.  Instead of building an
SDFG dataflow graph, it generates a *single* Python code string that becomes
the body of a ``TaskletNode`` inside a ``MapScope``.

The key prerequisite (confirmed in the design) is that the full shape of
input arguments (arrays) is passed to the tasklet, so ``deref`` does not
happen on the input edge but inside the tasklet as a Python array access.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import dace

from gt4py import eve
from gt4py.eve import codegen
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_types import (
    FieldopData,
)
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_utils import (
    _CONST_DIM,
    format_builtin,
)
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass
class CodegenContext:
    """Mutable context carried through the code-generation visit.

    Attributes:
        args_map: Maps lambda parameter names to their argument expressions.
            Updated when a let-lambda is encountered (dict-union = lexical
            shadowing).
        field_args: Maps lambda parameter names to ``FieldopData`` handles.
            Set once at the fieldop level and used to generate array-access
            expressions for ``deref``.
        map_indices: Maps ``Dimension`` to the corresponding map variable
            name (e.g. ``__i_Vertical_gtx_global``).  Used to generate
            array indices in the tasklet code.
        offset_provider_type: The offset provider type table, used for
            ``neighbors`` and unstructured shift.
        pre_statements: Accumulated Python statements that must appear
            before the final expression (e.g. ``for`` loops for ``reduce``).
        _temp_counter: Counter for generating unique temporary names.
    """

    args_map: dict[str, gtir.Node]
    field_args: dict[str, FieldopData]
    map_indices: dict[gtx_common.Dimension, str]
    offset_provider_type: gtx_common.OffsetProviderType
    pre_statements: list[str] = dataclasses.field(default_factory=list)
    _temp_counter: int = 0

    def fresh_temp(self) -> str:
        """Generate a unique temporary variable name."""
        self._temp_counter += 1
        return f"__stree_tmp_{self._temp_counter}"

    def child(self, **overrides: Any) -> CodegenContext:
        """Create a child context with the given overrides."""
        return dataclasses.replace(self, **overrides)


class StreePythonCodegen(eve.NodeVisitor):
    """Generate Python code for the stencil body of a field operator.

    The generated code is embedded in a ``TaskletNode`` inside a ``MapScope``.
    The full input arrays are passed to the tasklet, and ``deref`` is lowered
    to an array access inside the tasklet code.

    Usage::

        ctx = CodegenContext(args_map=..., field_args=..., map_indices=..., ...)
        expr_code = StreePythonCodegen().visit(stencil_expr, ctx=ctx)
        tasklet_code = "\\n".join(ctx.pre_statements) + f"\\n__out = {expr_code}"
    """

    Literal = codegen.FormatTemplate("{value}")

    def visit_Literal(self, node: gtir.Literal, *, ctx: CodegenContext) -> str:
        return str(node.value)

    def visit_SymRef(self, node: gtir.SymRef, *, ctx: CodegenContext) -> str:
        """Resolve a symbol reference.

        If the symbol is a lambda parameter (in ``args_map``), visit the
        mapped argument expression.  Otherwise, return the symbol name
        (it refers to an SDFG symbol or a data container).
        """
        symbol = str(node.id)
        if symbol in ctx.args_map:
            return self.visit(ctx.args_map[symbol], ctx=ctx)
        return symbol

    def visit_FunCall(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        # Handle let-lambdas: inline by updating args_map
        if isinstance(node.fun, gtir.Lambda):
            new_args_map = ctx.args_map | {
                p.id: arg for p, arg in zip(node.fun.params, node.args, strict=True)
            }
            return self.visit(node.fun.expr, ctx=ctx.child(args_map=new_args_map))

        # Handle deref
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node, ctx=ctx)

        # Handle neighbors
        if cpm.is_call_to(node, "neighbors"):
            return self._visit_neighbors(node, ctx=ctx)

        # Handle reduce (curried: reduce(op, init)(expr))
        if isinstance(node.fun, gtir.FunCall) and cpm.is_call_to(node.fun, "reduce"):
            return self._visit_reduce(node, ctx=ctx)

        # Handle map (curried: map(builtin)(args...))
        if isinstance(node.fun, gtir.FunCall) and cpm.is_call_to(node.fun, "map"):
            return self._visit_map(node, ctx=ctx)

        # Handle list_get
        if cpm.is_call_to(node, "list_get"):
            return self._visit_list_get(node, ctx=ctx)

        # Handle if_ (ternary)
        if cpm.is_call_to(node, "if_"):
            cond = self.visit(node.args[0], ctx=ctx)
            true_val = self.visit(node.args[1], ctx=ctx)
            false_val = self.visit(node.args[2], ctx=ctx)
            return format_builtin("if_", cond, true_val, false_val)

        # Handle make_tuple
        if cpm.is_call_to(node, "make_tuple"):
            args = [self.visit(arg, ctx=ctx) for arg in node.args]
            return f"({', '.join(args)})" if len(args) > 1 else f"({args[0]},)"

        # Handle generic builtins (math, cast_, tuple_get, etc.)
        if isinstance(node.fun, gtir.SymRef):
            builtin_name = str(node.fun.id)
            args = [self.visit(arg, ctx=ctx) for arg in node.args]
            return format_builtin(builtin_name, *args)

        raise NotImplementedError(f"Unexpected 'FunCall' node ({node}).")

    def _resolve_field_arg(
        self, node: gtir.SymRef, ctx: CodegenContext
    ) -> FieldopData | None:
        """Resolve a SymRef through args_map to a FieldopData.

        Returns ``None`` if the symbol does not resolve to a field argument
        (e.g., it's an SDFG symbol or a literal).
        """
        symbol = str(node.id)
        # Resolve through args_map
        while symbol in ctx.args_map:
            mapped = ctx.args_map[symbol]
            if isinstance(mapped, gtir.SymRef):
                symbol = str(mapped.id)
            else:
                return None
        return ctx.field_args.get(symbol)

    def _visit_deref(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``deref(expr)`` to a Python array-access expression.

        For a simple ``deref(SymRef("x"))`` where ``x`` is a field argument,
        generates ``x[i - origin, ...]`` using the map variables.

        For ``deref(shift(CartesianOffset(...), Literal("1"))(SymRef("x")))``,
        the shift offset is accumulated and added to the index.
        """
        arg = node.args[0]

        # Collect shifts by walking through curried shift calls
        shift_offsets: dict[gtx_common.Dimension, str] = {}
        while cpm.is_applied_shift(arg):
            shift_call = arg.fun  # FunCall(SymRef("shift"), [offset_args...])
            shift_args = shift_call.args
            assert len(shift_args) >= 2 and len(shift_args) % 2 == 0
            # Process each (offset_provider, offset_value) pair
            for i in range(0, len(shift_args), 2):
                offset_provider = shift_args[i]
                offset_value = shift_args[i + 1]
                if isinstance(offset_provider, gtir.CartesianOffset):
                    dim = gtx_common.Dimension(
                        offset_provider.domain.value,
                        gtx_common.DimensionKind.find(offset_provider.domain.value),
                    )
                    offset_str = self.visit(offset_value, ctx=ctx)
                    shift_offsets[dim] = offset_str
                else:
                    raise NotImplementedError(
                        f"Non-Cartesian shift '{offset_provider}' is not supported in stree lowering. "
                        "Use 'neighbors' for unstructured grids."
                    )
            arg = arg.args[0]  # the iterator

        assert isinstance(arg, gtir.SymRef), f"Unexpected deref argument: {arg}"
        field_data = self._resolve_field_arg(arg, ctx)

        if field_data is None:
            # Not a field argument — visit as a regular expression
            return self.visit(arg, ctx=ctx)

        if isinstance(field_data.gt_type, ts.ScalarType):
            # Scalar access (no index)
            return field_data.name

        # Array access: field_name[i - origin + shift, ...]
        assert isinstance(field_data.gt_type, ts.FieldType)
        indices = []
        for dim, origin in zip(field_data.gt_type.dims, field_data.origin, strict=True):
            map_var = ctx.map_indices.get(dim)
            if map_var is None:
                raise ValueError(f"No map variable for dimension {dim}.")
            index_expr = f"({map_var} - {origin})"
            if dim in shift_offsets:
                index_expr = f"({index_expr} + {shift_offsets[dim]})"
            indices.append(index_expr)
        return f"{field_data.name}[{', '.join(indices)}]"

    def _visit_neighbors(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``neighbors(SymRef("x"), offset)`` to a list comprehension.

        Generates:
        ``[x[gt_conn_{offset}[{source_map_var}, __n] - {origin}, ...] for __n in range({max_neighbors})]``

        With skip-value handling if the offset provider has skip values.
        """
        assert len(node.args) == 2
        field_arg = node.args[0]
        offset_arg = node.args[1]

        # Resolve the offset name
        assert isinstance(offset_arg, gtir.SymRef)
        offset_name = str(offset_arg.id)

        # Get the connectivity type
        offset_provider_type = ctx.offset_provider_type[offset_name]
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)

        # Resolve the field argument
        assert isinstance(field_arg, gtir.SymRef)
        field_data = self._resolve_field_arg(field_arg, ctx)
        assert field_data is not None, f"Field argument '{field_arg}' not found."
        assert isinstance(field_data.gt_type, ts.FieldType)

        # Get the connectivity table name and source dimension
        conn_name = gtx_dace_args.connectivity_identifier(offset_name)
        source_dim = offset_provider_type.source_dim
        source_map_var = ctx.map_indices.get(source_dim)
        assert source_map_var is not None, f"No map variable for source dimension {source_dim}."

        # Get max neighbors
        max_neighbors = offset_provider_type.max_neighbors

        # Generate the neighbor index variable
        neighbor_var = ctx.fresh_temp()

        # Build the array access for one neighbor
        indices = []
        for dim, origin in zip(field_data.gt_type.dims, field_data.origin, strict=True):
            if dim == source_dim:
                # Use the connectivity table for the source dimension
                indices.append(f"{conn_name}[{source_map_var}, {neighbor_var}] - {origin}")
            else:
                # Use the map variable for other dimensions
                map_var = ctx.map_indices.get(dim)
                assert map_var is not None, f"No map variable for dimension {dim}."
                indices.append(f"{map_var} - {origin}")

        access_expr = f"{field_data.name}[{', '.join(indices)}]"

        # Handle skip values
        if offset_provider_type.has_skip_values:
            skip_value = gtx_common._DEFAULT_SKIP_VALUE
            field_dtype = field_data.gt_type.dtype
            if isinstance(field_dtype, ts.ScalarType):
                dc_dtype = gtx_dace_args.as_dace_type(field_dtype)
            else:
                assert isinstance(field_dtype, ts.ListType)
                assert isinstance(field_dtype.element_type, ts.ScalarType)
                dc_dtype = gtx_dace_args.as_dace_type(field_dtype.element_type)
            # Use NaN for floating point, max value for integers
            import numpy as np

            if np.issubdtype(dc_dtype.as_numpy_dtype(), np.floating):
                replacement = "math.nan"
            else:
                replacement = str(dace.dtypes.max_value(dc_dtype))
            access_expr = (
                f"({access_expr}) if {conn_name}[{source_map_var}, {neighbor_var}] != {skip_value} "
                f"else {replacement}"
            )

        return f"[{access_expr} for {neighbor_var} in range({max_neighbors})]"

    def _visit_reduce(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``reduce(op, init)(expr)`` to a Python accumulation.

        Generates pre-statements:
        ``__reduce_N = init``
        ``for __n_N in {expr}:``
        ``    __reduce_N = {op}(__reduce_N, __n_N)``

        Returns ``__reduce_N``.
        """
        assert isinstance(node.fun, gtir.FunCall)
        assert cpm.is_call_to(node.fun, "reduce")
        assert len(node.fun.args) == 2
        assert len(node.args) == 1

        op_name = str(node.fun.args[0].id) if isinstance(node.fun.args[0], gtir.SymRef) else None
        assert op_name is not None, f"Expected SymRef as reduce operation, got {node.fun.args[0]}"
        init_expr = self.visit(node.fun.args[1], ctx=ctx)
        reduce_expr = self.visit(node.args[0], ctx=ctx)

        result_var = ctx.fresh_temp()
        loop_var = ctx.fresh_temp()

        ctx.pre_statements.append(f"{result_var} = {init_expr}")
        ctx.pre_statements.append(f"for {loop_var} in {reduce_expr}:")
        op_code = format_builtin(op_name, result_var, loop_var)
        ctx.pre_statements.append(f"    {result_var} = {op_code}")

        return result_var

    def _visit_map(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``map(builtin)(args...)`` to a Python list comprehension.

        Generates:
        ``[{builtin}(__arg0, __arg1, ...) for __n in range({local_size})]``
        """
        assert isinstance(node.fun, gtir.FunCall)
        assert cpm.is_call_to(node.fun, "map")
        assert len(node.fun.args) == 1  # the operation to be mapped

        op = node.fun.args[0]
        args = [self.visit(arg, ctx=ctx) for arg in node.args]

        # Find the local dimension size from the input args
        local_size = None
        for arg_node in node.args:
            arg_type = arg_node.type
            if isinstance(arg_type, ts.ListType) and arg_type.offset_type is not None:
                offset_type = arg_type.offset_type
                if offset_type == _CONST_DIM:
                    continue
                offset_provider_t = ctx.offset_provider_type[offset_type.value]
                assert isinstance(offset_provider_t, gtx_common.NeighborConnectivityType)
                local_size = offset_provider_t.max_neighbors
                break

        if local_size is None:
            raise ValueError(f"Missing information on local dimension for map node {node}.")

        loop_var = ctx.fresh_temp()

        # Generate the mapped expression
        if isinstance(op, gtir.SymRef):
            op_name = str(op.id)
            mapped_args = []
            for _i, arg in enumerate(args):
                # If the arg is a list, index it by the loop variable
                # Otherwise, use it as-is (broadcast)
                if f"[{loop_var}]" in arg or f"for {loop_var}" in arg:
                    mapped_args.append(f"{arg}[{loop_var}]" if not arg.startswith("[") else arg)
                else:
                    mapped_args.append(arg)
            mapped_expr = format_builtin(op_name, *mapped_args)
        else:
            raise NotImplementedError(f"Non-SymRef map operation not supported: {op}")

        return f"[{mapped_expr} for {loop_var} in range({local_size})]"

    def _visit_list_get(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``list_get(index, expr)`` to ``expr[index]``."""
        assert len(node.args) == 2
        index = self.visit(node.args[0], ctx=ctx)
        expr = self.visit(node.args[1], ctx=ctx)
        return f"{expr}[{index}]"


def generate_tasklet_code(
    expr: gtir.Expr,
    *,
    field_args: dict[str, FieldopData],
    map_indices: dict[gtx_common.Dimension, str],
    offset_provider_type: gtx_common.OffsetProviderType,
    args_map: dict[str, gtir.Node] | None = None,
) -> tuple[str, list[str]]:
    """Generate the Python code for a tasklet body.

    Args:
        expr: The GTIR expression (stencil body) to lower.
        field_args: Mapping from lambda parameter names to ``FieldopData``.
        map_indices: Mapping from ``Dimension`` to map variable name.
        offset_provider_type: The offset provider type table.
        args_map: Initial args_map for let-lambda inlining (usually empty).

    Returns:
        A tuple ``(expr_code, pre_statements)`` where ``expr_code`` is the
        final Python expression and ``pre_statements`` is a list of Python
        statements that must appear before the final expression.  The caller
        should join them as: ``"\\n".join(pre_statements) + f"\\n__out = {expr_code}"``.
    """
    ctx = CodegenContext(
        args_map=args_map or {},
        field_args=field_args,
        map_indices=map_indices,
        offset_provider_type=offset_provider_type,
    )
    expr_code = StreePythonCodegen().visit(expr, ctx=ctx)
    return expr_code, ctx.pre_statements
