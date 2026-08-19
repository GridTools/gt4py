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
from gt4py.next.iterator import builtins as itir_builtins, ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, misc as itir_misc
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
        used_connectivities: Names of connectivity tables (offset names)
            referenced by ``neighbors`` in the stencil body.  Populated
            during the visit so the caller can add them as tasklet inputs;
            otherwise the connectivity arrays would be dangling references
            inside the tasklet code.
        _temp_counter: Counter for generating unique temporary names.
    """

    args_map: dict[str, gtir.Node]
    field_args: dict[str, FieldopData]
    map_indices: dict[gtx_common.Dimension, str]
    offset_provider_type: gtx_common.OffsetProviderType
    pre_statements: list[str] = dataclasses.field(default_factory=list)
    used_connectivities: set[str] = dataclasses.field(default_factory=set)
    string_args: dict[str, str] = dataclasses.field(default_factory=dict)
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

    def visit_OffsetLiteral(self, node: gtir.OffsetLiteral, *, ctx: CodegenContext) -> str:
        return str(node.value)

    def visit_SymRef(self, node: gtir.SymRef, *, ctx: CodegenContext) -> str:
        """Resolve a symbol reference.

        If the symbol is a lambda parameter (in ``args_map``), visit the
        mapped argument expression.  Otherwise, return the symbol name
        (it refers to an SDFG symbol or a data container).
        """
        symid = str(node.id)
        if symid in ctx.field_args:
            return symid
        elif symid in ctx.string_args:
            return ctx.string_args[symid]
        elif symid in ctx.args_map:
            return self.visit(ctx.args_map[symid], ctx=ctx)
        elif symid in itir_builtins.TYPE_BUILTINS:
            return symid
        raise ValueError(f"Unknown symbol '{symid}' in node '{node}'.")

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

        # Handle make_const_list: broadcast value, no indexing
        if cpm.is_call_to(node, "make_const_list"):
            assert len(node.args) == 1
            return self.visit(node.args[0], ctx=ctx)

        # Handle reduce (curried: reduce(op, init)(expr))
        if cpm.is_call_to(node.fun, "reduce"):
            return self._visit_reduce(node, ctx=ctx)

        # Handle map (curried: map(builtin)(args...))
        if cpm.is_call_to(node.fun, "map"):
            return self._visit_map(node, ctx=ctx)

        # Handle map_list (curried: map_list(op)(args...))
        if cpm.is_call_to(node.fun, "map_list"):
            return self._visit_map_list(node, ctx=ctx)

        # Handle list_get
        if cpm.is_call_to(node, "list_get"):
            return self._visit_list_get(node, ctx=ctx)

        # Handle if_ (ternary)
        if cpm.is_call_to(node, "if_"):
            cond = self.visit(node.args[0], ctx=ctx)
            true_val = self.visit(node.args[1], ctx=ctx)
            false_val = self.visit(node.args[2], ctx=ctx)
            code = format_builtin("if_", cond, true_val, false_val)
            return f"({code})"

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
        """Resolve a SymRef (possibly through args_map) to its FieldopData.

        The symbol is first looked up directly in ``field_args`` (the common
        case: it is a field-operator lambda parameter).  If not found, the
        ``args_map`` chain — updated when let-lambdas are encountered in
        ``visit_FunCall`` — is followed until a symbol in ``field_args`` is
        reached.

        Returns ``None`` if the symbol does not resolve to a field argument
        (e.g. it refers to an SDFG data container or a symbol); callers fall
        back to ``visit_SymRef`` in that case.
        """
        symbol = str(node.id)
        if symbol in ctx.field_args:
            return ctx.field_args[symbol]

        seen: set[str] = {symbol}
        while symbol in ctx.args_map:
            mapped = ctx.args_map[symbol]
            if isinstance(mapped, gtir.SymRef):
                symbol = str(mapped.id)
                if symbol in ctx.field_args:
                    return ctx.field_args[symbol]
                if symbol in seen:
                    break
                seen.add(symbol)
            else:
                break

        return None

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
                offset = shift_args[i]
                offset_value = shift_args[i + 1]
                if isinstance(offset, gtir.CartesianOffset):
                    dim = itir_misc.dim_from_axis_literal(offset.codomain)
                    offset_str = self.visit(offset_value, ctx=ctx)
                    shift_offsets[dim] = offset_str
                else:
                    raise NotImplementedError(
                        f"Non-Cartesian shift '{offset}' is not supported in stree lowering. "
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

    def _single_neighbor_access(
        self, node: gtir.FunCall, neighbor_var: str, ctx: CodegenContext
    ) -> tuple[str, str, str]:
        """Build the scalar access expression for one neighbor.

        Args:
            node: A ``neighbors(offset, source)`` ``FunCall``.
            neighbor_var: The C variable used to index the neighbor
                (i.e. the connectivity table's second axis).
            ctx: The codegen context.

        Returns:
            A ``(access_expr, conn_name, source_map_var)`` tuple where
            ``access_expr`` is a Python expression yielding the value of one
            neighbor, ``conn_name`` is the connectivity container name (for
            skip-value checks), and ``source_map_var`` is the map variable of
            the source dimension.  The ``conn_name`` is also recorded in
            ``ctx.used_connectivities``.
        """
        assert len(node.args) == 2
        offset = node.args[0]
        source = node.args[1]

        assert isinstance(offset, gtir.OffsetLiteral)
        assert isinstance(offset.value, str), f"Expected string offset name, got '{offset.value}'"

        offset_type = ctx.offset_provider_type[offset.value]
        assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
        ctx.used_connectivities.add(gtx_dace_args.connectivity_identifier(offset.value))

        assert isinstance(source, gtir.SymRef)
        field_data = self._resolve_field_arg(source, ctx)
        assert field_data is not None, f"Field argument '{source}' not found."
        assert isinstance(field_data.gt_type, ts.FieldType)

        conn_name = gtx_dace_args.connectivity_identifier(offset.value)
        source_dim = offset_type.source_dim
        source_map_var = ctx.map_indices.get(source_dim)
        assert source_map_var is not None, f"No map variable for source dimension {source_dim}."

        # Build the array access for one neighbor.  The field's dimension
        # that matches the connectivity's codomain is indexed through the
        # connectivity table; all other dimensions use their map variable.
        indices = []
        for dim, origin in zip(field_data.gt_type.dims, field_data.origin, strict=True):
            if dim == offset_type.codomain:
                indices.append(f"{conn_name}[{source_map_var}, {neighbor_var}] - {origin}")
            else:
                map_var = ctx.map_indices.get(dim)
                assert map_var is not None, f"No map variable for dimension {dim}."
                indices.append(f"{map_var} - {origin}")

        access_expr = f"{field_data.name}[{', '.join(indices)}]"

        # Handle skip values
        if offset_type.has_skip_values:
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

        return access_expr, conn_name, source_map_var

    def _visit_neighbors(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``neighbors(SymRef("x"), offset)`` to a local list.

        Generates pre-statements that materialize the neighbor values into a
        local list ``__neigh_N`` via a regular ``for`` loop:

        ``__neigh_N = [0] * {max_neighbors}``
        ``for __n_N in range({max_neighbors}):``
        ``    __neigh_N[__n_N] = x[gt_conn_{offset}[{source_map_var}, __n_N] - {origin}, ...]``

        With skip-value handling, the assignment becomes a conditional
        expression substituting the skip replacement for invalid neighbors.

        A regular loop (rather than a list comprehension) is used because
        DaCe's Python free-symbol analysis does not recognize list-comprehension
        targets as bound, which would leave the neighbor variable as a
        dangling free symbol in the tasklet.

        Note:
            DaCe transpiles Python tasklets to C++, which does not support
            list creation.  This lowering therefore only works when the
            ``neighbors`` expression is consumed by ``reduce`` (see
            ``_visit_reduce``, which fuses the two and avoids the list).
            Standalone ``neighbors`` requires a different lowering strategy.
        """
        offset_type = ctx.offset_provider_type[node.args[0].value]  # type: ignore[index]
        assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
        max_neighbors = offset_type.max_neighbors

        neighbor_var = ctx.fresh_temp()
        result_var = ctx.fresh_temp()
        access_expr, _conn_name, _source_map_var = self._single_neighbor_access(
            node, neighbor_var, ctx
        )

        ctx.pre_statements.append(f"{result_var} = [0] * {max_neighbors}")
        ctx.pre_statements.append(f"for {neighbor_var} in range({max_neighbors}):")
        ctx.pre_statements.append(f"    {result_var}[{neighbor_var}] = {access_expr}")
        return result_var

    def _visit_reduce(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``reduce(op, init)(expr)`` to a Python accumulation.

        For the common ``reduce(op, init)(neighbors(...))`` shape the two are
        fused into a single scalar accumulation loop over the connectivity
        table, which avoids creating a Python list (DaCe's Python-to-C++
        transpiler cannot handle list operations)::

            __reduce_N = init
            for __n_N in range({max_neighbors}):
                __reduce_N = {op}(__reduce_N, {single_neighbor_access})

        For other expressions the reduce iterates over ``expr`` as a Python
        iterable.

        Returns:
            The name of the variable holding the reduction result.
        """
        assert isinstance(node.fun, gtir.FunCall)
        assert cpm.is_call_to(node.fun, "reduce")
        assert len(node.fun.args) == 2
        assert len(node.args) == 1

        op_name = str(node.fun.args[0].id) if isinstance(node.fun.args[0], gtir.SymRef) else None
        assert op_name is not None, f"Expected SymRef as reduce operation, got {node.fun.args[0]}"
        init_expr = self.visit(node.fun.args[1], ctx=ctx)

        result_var = ctx.fresh_temp()
        ctx.pre_statements.append(f"{result_var} = {init_expr}")

        inner = node.args[0]

        # --- Fuse reduce + neighbors --------------------------------------
        if cpm.is_call_to(inner, "neighbors"):
            # Fuse reduce + neighbors into a scalar loop over the
            # connectivity table (no Python list allocation).
            offset_type = ctx.offset_provider_type[inner.args[0].value]  # type: ignore[index]
            assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
            loop_var = ctx.fresh_temp()
            access_expr, _conn_name, _source_map_var = self._single_neighbor_access(
                inner, loop_var, ctx
            )
            ctx.pre_statements.append(f"for {loop_var} in range({offset_type.max_neighbors}):")
            ctx.pre_statements.append(
                f"    {result_var} = {format_builtin(op_name, result_var, access_expr)}"
            )

        # --- Fuse reduce + map_list ---------------------------------------
        elif cpm.is_call_to(inner.fun, "map_list"):
            # Fuse reduce + map_list into a scalar accumulation loop
            # (init; for n in range(N): result = reduce_op(result,
            # map_op(elem_0, elem_1, ...))) to avoid Python list creation.
            map_op = inner.fun.args[0]  # type: ignore[union-attr]
            map_args = inner.args

            # Determine local size from the first non-const argument.
            local_size = None
            for arg_node in map_args:
                _expr, sz = self._single_map_element(arg_node, "__dummy", ctx)
                if sz is not None:
                    local_size = sz
                    break
            if local_size is None:
                raise ValueError(
                    f"Missing information on local dimension for map_list in {node}."
                )

            loop_var = ctx.fresh_temp()
            # Compute the element expression for each map_list argument.
            mapped_args = [
                self._single_map_element(arg, loop_var, ctx)[0] for arg in map_args
            ]

            # Build the mapped expression: map_op(elem_0, elem_1, ...)
            if isinstance(map_op, gtir.SymRef):
                mapped_expr = format_builtin(str(map_op.id), *mapped_args)
            elif isinstance(map_op, gtir.Lambda):
                # Bind lambda params to the mapped argument expressions
                # (as strings) and visit the body.
                lambda_ctx = ctx.child(
                    string_args=ctx.string_args
                    | {str(p.id): a for p, a in zip(map_op.params, mapped_args, strict=True)}
                )
                mapped_expr = self.visit(map_op.expr, ctx=lambda_ctx)
            else:
                raise NotImplementedError(
                    f"Non-SymRef/Lambda map_list operation not supported: {map_op}"
                )

            ctx.pre_statements.extend([
                f"for {loop_var} in range({local_size}):",
                f"    {result_var} = {format_builtin(op_name, result_var, mapped_expr)}"
            ])

        else:
            loop_var = ctx.fresh_temp()
            reduce_expr = self.visit(inner, ctx=ctx)
            ctx.pre_statements.extend([
                f"for {loop_var} in {reduce_expr}:",
                f"    {result_var} = {format_builtin(op_name, result_var, loop_var)}"
            ])

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

    def _single_map_element(
        self, arg: gtir.Node, loop_var: str, ctx: CodegenContext
    ) -> tuple[str, int | None]:
        """Generate the Python expression for one element of a ``map_list`` argument.

        Args:
            arg: An argument to ``map_list`` — typically ``neighbors(...)``,
                ``make_const_list(...)``, or a ``SymRef`` / ``deref`` of a
                local field.
            loop_var: The C variable used to index the local (neighbor)
                dimension.
            ctx: The codegen context.

        Returns:
            A ``(element_expr, max_neighbors)`` tuple where ``element_expr``
            is the Python expression for one element and ``max_neighbors``
            is the size of the local dimension (or ``None`` for const lists).
        """
        # Case 1: neighbors(offset, source) — single neighbor access
        if cpm.is_call_to(arg, "neighbors"):
            access_expr, _conn_name, _source_map_var = self._single_neighbor_access(
                arg, loop_var, ctx
            )
            offset_type = ctx.offset_provider_type[arg.args[0].value]  # type: ignore[index]
            assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
            return access_expr, offset_type.max_neighbors

        # Case 2: make_const_list(value) — broadcast, no indexing
        if cpm.is_call_to(arg, "make_const_list"):
            assert len(arg.args) == 1
            return self.visit(arg.args[0], ctx=ctx), None

        # Case 3: deref(SymRef) or SymRef to a local field — index by
        # map vars + loop var.  FuseAsFieldOp wraps local fields in
        # ``deref`` before passing them to ``map_list``.
        symref = arg
        if cpm.is_call_to(arg, "deref") and isinstance(arg.args[0], gtir.SymRef):
            symref = arg.args[0]
        if isinstance(symref, gtir.SymRef):
            field_data = self._resolve_field_arg(symref, ctx)
            if (
                field_data is not None
                and isinstance(field_data.gt_type, ts.FieldType)
                and isinstance(field_data.gt_type.dtype, ts.ListType)
            ):
                # Local field: index non-local dims by map vars, local dim
                # (with origin 0) by the loop variable.
                indices = []
                for dim, origin in zip(
                    field_data.gt_type.dims, field_data.origin, strict=True
                ):
                    map_var = ctx.map_indices.get(dim)
                    assert map_var is not None, f"No map variable for dimension {dim}."
                    indices.append(f"{map_var} - {origin}")
                indices.append(loop_var)
                local_dim = field_data.gt_type.dtype.offset_type
                assert local_dim is not None
                offset_provider_t = ctx.offset_provider_type.get(local_dim.value)
                max_neighbors = (
                    offset_provider_t.max_neighbors
                    if isinstance(offset_provider_t, gtx_common.NeighborConnectivityType)
                    else None
                )
                return f"{field_data.name}[{', '.join(indices)}]", max_neighbors

            if field_data is not None and isinstance(field_data.gt_type, ts.ScalarType):
                # Scalar: no indexing
                return field_data.name, None

        # Case 4: generic — visit and index by loop variable
        return f"{self.visit(arg, ctx=ctx)}[{loop_var}]", None

    def _visit_map_list(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``map_list(op)(args...)``.

        Note:
            DaCe's Python-to-C++ transpiler cannot handle Python list
            creation, so standalone ``map_list`` (not consumed by
            ``reduce``) is not supported.  When ``map_list`` is consumed by
            ``reduce``, the two are fused in :meth:`_visit_reduce` into a
            scalar accumulation loop that avoids lists entirely.
        """
        raise NotImplementedError(
            "Standalone 'map_list' is not supported in stree lowering; "
            "only 'reduce(...)(map_list(...)(...))' is supported."
        )

    def _visit_list_get(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``list_get(index, expr)`` to a scalar expression.

        When ``expr`` is ``map_list(op)(args...)``, the two are fused into
        ``op(args[0][index], args[1][index], ...)`` to avoid creating a
        Python list (which DaCe's Python-to-C++ transpiler cannot handle).

        When ``expr`` is ``neighbors(offset, source)``, the two are fused
        into a single neighbor access at the given index.

        Otherwise, lowers to ``expr[index]``.
        """
        assert len(node.args) == 2
        index = self.visit(node.args[0], ctx=ctx)
        inner = node.args[1]

        # Fuse list_get + map_list: instead of materializing the full list
        # and then indexing one element, evaluate the map operation on
        # just the element at the given index.
        if isinstance(inner, gtir.FunCall) and cpm.is_call_to(inner.fun, "map_list"):
            map_op = inner.fun.args[0]
            map_args = inner.args

            # Evaluate each argument at the given index.
            indexed_args = [
                self._single_map_element(arg, index, ctx)[0]
                for arg in map_args
            ]

            # Apply the map operation to the indexed arguments.
            if isinstance(map_op, gtir.SymRef):
                return format_builtin(str(map_op.id), *indexed_args)
            elif isinstance(map_op, gtir.Lambda):
                lambda_ctx = ctx.child(
                    string_args=ctx.string_args
                    | {str(p.id): a for p, a in zip(map_op.params, indexed_args, strict=True)}
                )
                return self.visit(map_op.expr, ctx=lambda_ctx)
            else:
                raise NotImplementedError(
                    f"Non-SymRef/Lambda map_list operation not supported: {map_op}"
                )

        # Fuse list_get + neighbors: single neighbor access at the index.
        if cpm.is_call_to(inner, "neighbors"):
            access_expr, _conn_name, _source_map_var = self._single_neighbor_access(
                inner, index, ctx
            )
            return access_expr

        return f"{self.visit(inner, ctx=ctx)}[{index}]"


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
        A tuple ``(expr_code, pre_statements, used_connectivities)`` where
        ``expr_code`` is the final Python expression, ``pre_statements`` is a
        list of Python statements that must appear before the final
        expression (join them as: ``"\\n".join(pre_statements) +
        f"\\n__out = {expr_code}"``), and ``used_connectivities`` is the set
        of connectivity-table names (offset names) referenced by
        ``neighbors`` in the stencil body.
    """
    ctx = CodegenContext(
        args_map=args_map or {},
        field_args=field_args,
        map_indices=map_indices,
        offset_provider_type=offset_provider_type,
    )
    expr_code = StreePythonCodegen().visit(expr, ctx=ctx)
    return expr_code, ctx.pre_statements, ctx.used_connectivities
