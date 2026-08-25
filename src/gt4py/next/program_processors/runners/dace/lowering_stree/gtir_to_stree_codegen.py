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

``deref`` is lowered the same way as in ``lowering.gtir_to_sdfg_iterator``:
for each dimension, the index is classified as *symbolic* (compile-time
known — no shift or cartesian shift with literal offset) or *dynamic*
(runtime value — cartesian shift with dynamic offset).  Symbolic dimensions
are indexed on the input edge to the tasklet (memlet subset), so the
tasklet receives a scalar connector.  When some dimension is *dynamic*,
the full array is passed to the tasklet and the multi-dimensional access
is emitted on the tasklet-connector name (see ``_access_name``): DaCe's
Python-to-C++ transpiler linearizes connector-based multi-dimensional
access using the container strides, while a multi-dimensional subscript
on a raw container name would not be translatable.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Mapping

import dace
import numpy as np

from gt4py import eve
from gt4py.eve import codegen
from gt4py.next import common as gtx_common
from gt4py.next.iterator import builtins as itir_builtins, ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as itir_misc,
)
from gt4py.next.iterator.type_system import type_specifications as itir_ts
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_types import DataRef
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_utils import (
    _CONST_DIM,
    format_builtin,
    make_symbol_tree,
)
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class ListElementAccess:
    """Scalar access to one element of a neighbor list in stree codegen.

    Attributes:
        expr: Python expression for the element.
        mask: Skip-value guard condition (``conn[i, n] != skip_value``) when
            the element is read through a connectivity with skip values,
            else ``None``.  Fused reductions use it as a statement-level
            ``if`` guard around the accumulation (matching
            ``ReduceWithSkipValues`` in the SDFG lowering, which excludes
            skipped neighbors regardless of how the element expression is
            composed); list materializations embed it in a conditional
            expression.
        dummy: Replacement value for skipped elements when the neighbor
            values are materialized into a list — ``math.nan`` for
            floating-point, the dtype's max value for integers (matching
            the SDFG lowering, where the dummy is masked out by any
            skip-aware consumer).
        body: Statements evaluating ``expr`` — non-empty only for
            ``neighbors(offset, (↑stencil)(...))`` sources, where the
            stencil body (e.g. containing a nested ``reduce``) is evaluated
            per neighbor and generates its own accumulation loop (see
            ``_single_lifted_map_element``).  Callers must emit these at
            the element-evaluation site, guarded by ``mask`` when present —
            the inner loop reads the connectivity table at the neighbor
            position, which is out of bounds for a skipped neighbor.
    """

    expr: str
    mask: str | None = None
    dummy: str = ""
    body: tuple[str, ...] = ()


@dataclasses.dataclass
class CodegenContext:
    """Mutable context carried through the code-generation visit.

    Attributes:
        args_map: Maps lambda parameter names to their argument expressions.
            Updated when a let-lambda is encountered (dict-union = lexical
            shadowing).
        data_args: Maps lambda parameter names to ``DataRef`` handles.
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
        field_inputs: Maps connector names to ``(field_name, subset)``
            pairs for scalar field accesses that should be materialised on
            the input edge (memlet) rather than inside the tasklet code.
            Populated by ``_visit_deref`` for cartesian shift with literal
            offset (and direct deref without shift), where the full index
            expression is compile-time known and can be encoded in a
            memlet subset.  This avoids multi-dimensional array access
            (``a[i, j, k]``) inside the tasklet, which DaCe's Python-to-C++
            transpiler cannot lower.
        dynamic_map_indices: Dimensions whose ``map_indices`` entry is a
            runtime index expression rather than a map variable — e.g. the
            neighbor position ``gt_conn_X[i, n]`` injected by
            ``_single_lifted_map_element`` to evaluate a nested reduce per
            neighbor.  Such indices classify as dynamic in ``deref`` (they
            read the connectivity table, which is not available on the
            memlet edge).
        scalar_field_dims: For each field container name, the dimensions
            whose container extent is statically one (scalar dimensions).
            Such dimensions are squeezed out of tasklet subscripts: DaCe
            requires the subscript dimensionality of a tasklet connector
            to match the number of non-scalar (non-singleton) dimensions
            of the memlet subset (``ConnectorDimensionalityValidator``).
        _temp_counter: Counter for generating unique temporary names.
    """

    args_map: dict[str, gtir.Node]
    data_args: dict[str, DataRef]
    map_indices: dict[gtx_common.Dimension, str]
    offset_provider_type: gtx_common.OffsetProviderType
    dynamic_map_indices: frozenset[gtx_common.Dimension] = frozenset()
    #: Names of SDFG-level symbols and data containers; a ``SymRef``
    #: resolving to one of these is emitted literally (they are free symbols
    #: inside the tasklet code).
    root_symbols: frozenset[str] = frozenset()
    pre_statements: list[str] = dataclasses.field(default_factory=list)
    used_connectivities: set[str] = dataclasses.field(default_factory=set)
    string_args: dict[str, str] = dataclasses.field(default_factory=dict)
    field_inputs: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)
    scalar_field_dims: dict[str, frozenset[gtx_common.Dimension]] = dataclasses.field(
        default_factory=dict
    )
    is_inner_expr: bool = False
    # Shared monotonic counter; an iterator (rather than an ``int``) so that
    # ``child()`` contexts (created via ``dataclasses.replace``, which would
    # copy a plain counter and lead to duplicate names after inlining
    # let-lambdas) share the counter state.
    _temp_counter: itertools.count[int] = dataclasses.field(
        default_factory=lambda: itertools.count(1)
    )

    def fresh_temp(self) -> str:
        """Generate a unique temporary variable name."""
        return f"__stree_tmp_{next(self._temp_counter)}"

    def fresh_deref_connector(self) -> str:
        """Generate a unique connector name for a scalar field access."""
        return f"__deref_{next(self._temp_counter)}"

    def child(self, **overrides: Any) -> CodegenContext:
        """Create a child context with the given overrides."""
        return dataclasses.replace(self, **overrides)

    def bind_args(self, new_bindings: Mapping[Any, gtir.Node]) -> CodegenContext:
        """Bind new lambda arguments in a child context.

        The bindings are merged into ``args_map`` (lexical shadowing); names
        that are also present in ``data_args`` are removed from there, so
        that an inner lambda parameter shadowing a field-operator parameter
        (e.g. from a fused-in lambda) resolves to the inner binding and not
        to the outer field.
        """
        bound_names = {str(k) for k in new_bindings}
        args_map = self.args_map | {str(k): v for k, v in new_bindings.items()}
        data_args = {name: data for name, data in self.data_args.items() if name not in bound_names}
        return self.child(args_map=args_map, data_args=data_args)

    def bind_strings(self, new_bindings: Mapping[str, str]) -> CodegenContext:
        """Bind new string-valued arguments in a child context.

        Same scoping rule as ``bind_args`` — the bound names are removed from
        ``data_args`` so that they shadow field-operator parameters.
        """
        bound_names = {str(k) for k in new_bindings}
        string_args = self.string_args | {str(k): v for k, v in new_bindings.items()}
        data_args = {name: data for name, data in self.data_args.items() if name not in bound_names}
        return self.child(string_args=string_args, data_args=data_args)


@dataclasses.dataclass(frozen=True)
class ShiftReplacement:
    """Base class for a shift replacement applied to an index dimension.

    A shift replaces ``old_dim`` (a dimension in the field-operator domain)
    with ``new_dim`` (a dimension in the field), adjusting the index.
    ``is_symbolic`` indicates whether the new index is compile-time known.
    """

    new_dim: gtx_common.Dimension
    is_symbolic: bool

    def apply(self, old_index: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class CartesianShiftReplacement(ShiftReplacement):
    """Cartesian shift: ``new_index = old_index + offset_str``."""

    offset_str: str

    def apply(self, old_index: str) -> str:
        if self.offset_str == "0":
            return old_index
        return f"({old_index} + {self.offset_str})"


def connectivity_table_index(max_neighbors: int, conn_name: str, row: str, col: str) -> str:
    """Subscript a connectivity table inside tasklet code.

    When ``max_neighbors == 1``, the neighbor dimension of the table has
    extent 1 in the input memlet; DaCe squeezes extent-1 memlet dims in
    the tasklet's connector type, so the neighbor index must be omitted
    from the subscript.
    """
    if max_neighbors == 1:
        return f"{conn_name}[{row}]"
    return f"{conn_name}[{row}, {col}]"


@dataclasses.dataclass(frozen=True)
class UnstructuredShiftReplacement(ShiftReplacement):
    """Unstructured shift: ``new_index = conn_name[old_index, offset_str]``.

    The connectivity table is passed to the tasklet as a full-array input;
    ``conn_name`` is also recorded in ``ctx.used_connectivities``.
    """

    conn_name: str
    offset_str: str
    max_neighbors: int

    def apply(self, old_index: str) -> str:
        return connectivity_table_index(
            self.max_neighbors, self.conn_name, old_index, self.offset_str
        )


def _sym_tree_to_make_tuple(tree: object) -> gtir.Expr:
    """Convert a nested tree of ``gtir.Sym`` (from ``make_symbol_tree``) into
    a nested ``make_tuple`` GTIR expression of ``SymRef`` leaves."""
    if isinstance(tree, tuple):
        return gtir.FunCall(
            fun=gtir.SymRef(id="make_tuple"),
            args=[_sym_tree_to_make_tuple(sub) for sub in tree],
        )
    assert isinstance(tree, gtir.Sym)
    return gtir.SymRef(id=tree.id, type=tree.type)


class StreePythonCodegen(eve.NodeVisitor):
    """Generate Python code for the stencil body of a field operator.

    The generated code is embedded in a ``TaskletNode`` inside a ``MapScope``.
    For ``deref`` with no shift or with cartesian shift with literal offset,
    the field access is done on the input edge to the tasklet — the tasklet
    receives the value through a scalar connector rather than indexing the
    full array inside the tasklet code.  For dynamic offsets, the full array
    is passed to the tasklet and ``deref`` indexes it inside the tasklet code.

    Usage::

        ctx = CodegenContext(args_map=..., data_args=..., map_indices=..., ...)
        expr_code = StreePythonCodegen().visit(stencil_expr, ctx=ctx)
        tasklet_code = "\\n".join(ctx.pre_statements) + f"\\n__out = {expr_code}"
    """

    Literal = codegen.FormatTemplate("{value}")

    def visit_Literal(self, node: gtir.Literal, *, ctx: CodegenContext) -> str:
        return str(node.value)

    def visit_OffsetLiteral(self, node: gtir.OffsetLiteral, *, ctx: CodegenContext) -> str:
        return str(node.value)

    def _bind_inlined_args(
        self, new_bindings: Mapping[Any, gtir.Expr], ctx: CodegenContext
    ) -> CodegenContext:
        """Bind the arguments of an inlined (let-/applied) lambda in a child context.

        Bindings to ``deref`` of a field argument are evaluated eagerly —
        that is, the indexed field access is generated now and the parameter
        is bound to the resulting expression via ``string_args``.  This is
        needed because such a binding is self-referential once the parameter
        name shadows the field argument in the child context (e.g. from CSE,
        ``(λ(b_) → ...)(·b_)`` binds ``b_`` to ``deref(b_)``): resolving the
        dereffed access lazily in the child context would loop through
        ``args_map``.  All other bindings stay lazy (call-by-name) in
        ``args_map``, as before.
        """
        args_bindings: dict[str, gtir.Expr] = {}
        string_bindings: dict[str, str] = {}
        for name, value in new_bindings.items():
            if cpm.is_call_to(value, "deref"):
                deref_arg = self._resolve_args_map_symbol(value.args[0], ctx)
                if isinstance(deref_arg, gtir.SymRef) and str(deref_arg.id) in ctx.data_args:
                    string_bindings[str(name)] = self._visit_deref(value, ctx=ctx)
                    continue
            args_bindings[str(name)] = value
        child_ctx = ctx.bind_args(args_bindings) if args_bindings else ctx
        if string_bindings:
            # The eagerly-resolved derefs replace the field arguments:
            # remove them so the string bindings take effect.
            child_ctx = child_ctx.child(
                data_args={
                    name: data
                    for name, data in child_ctx.data_args.items()
                    if name not in string_bindings
                }
            )
            child_ctx = child_ctx.bind_strings(string_bindings)
        return child_ctx

    def visit_SymRef(self, node: gtir.SymRef, *, ctx: CodegenContext) -> str:
        """Resolve a symbol reference.

        If the symbol is a lambda parameter (in ``args_map``), visit the
        mapped argument expression.  Otherwise, return the symbol name
        (it refers to an SDFG symbol or a data container).
        """
        symid = str(node.id)
        if symid in ctx.data_args:
            return symid
        elif symid in ctx.string_args:
            return ctx.string_args[symid]
        elif symid in ctx.args_map:
            return self.visit(ctx.args_map[symid], ctx=ctx)
        elif symid in itir_builtins.TYPE_BUILTINS:
            return symid
        elif symid in ctx.root_symbols:
            # SDFG symbol or data container: emitted literally (free symbol
            # inside the tasklet code).
            return symid
        raise ValueError(f"Unknown symbol '{symid}' in node '{node}'.")

    def visit_FunCall(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        # Handle let-lambdas: inline by updating args_map
        if cpm.is_let(node):
            new_bindings = {p.id: arg for p, arg in zip(node.fun.params, node.args, strict=True)}
            return self.visit(node.fun.expr, ctx=self._bind_inlined_args(new_bindings, ctx))

        # Handle lambdas bound to a symbol (via an outer let-lambda) applied
        # as functions — e.g. `FuseAsFieldOp` emits `_cs_0(_cs_4)` where
        # `_cs_0` binds the inner field-operator lambda.  Inline the lambda
        # body with its parameters bound to the call arguments.
        if isinstance(node.fun, gtir.SymRef) and str(node.fun.id) in ctx.args_map:
            mapped_fun = ctx.args_map[str(node.fun.id)]
            if isinstance(mapped_fun, gtir.Lambda):
                new_bindings = {
                    p.id: arg for p, arg in zip(mapped_fun.params, node.args, strict=True)
                }
                return self.visit(mapped_fun.expr, ctx=self._bind_inlined_args(new_bindings, ctx))

        # Handle applied lifted lambdas: `↑(λ(a_, ...) → body)(args...)` —
        # inline the lambda body with its parameters bound to the arguments.
        # This appears when a reduce field operator is fused into another
        # field operator: `deref(↑(λ(a_) → reduce(...))(a))` evaluates the
        # lifted lambda body at the current location.
        if cpm.is_applied_lift(node):
            lifted_lambda = self._resolve_lifted_lambda(node, ctx)
            new_bindings = {
                p.id: arg for p, arg in zip(lifted_lambda.params, node.args, strict=True)
            }
            return self.visit(lifted_lambda.expr, ctx=self._bind_inlined_args(new_bindings, ctx))

        # Handle deref
        if cpm.is_call_to(node, "deref"):
            return self._visit_deref(node, ctx=ctx)

        # Handle neighbors
        if cpm.is_call_to(node, "neighbors"):
            return self._visit_neighbors(node, ctx=ctx)

        # Handle can_deref
        if cpm.is_call_to(node, "can_deref"):
            return self._visit_can_deref(node, ctx=ctx)

        # Handle make_const_list: broadcast value, no indexing
        if cpm.is_call_to(node, "make_const_list"):
            assert len(node.args) == 1
            return self.visit(node.args[0], ctx=ctx)

        # Handle reduce (curried: reduce(op, init)(expr))
        if cpm.is_applied_reduce(node):
            return self._visit_reduce(node, ctx=ctx)

        # Handle map_list (curried: map_list(op)(args...))
        if cpm.is_applied_map(node):
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

        # Handle tuple_get on tuples in the stencil body — e.g. produced
        # when a tuple-output `as_fieldop` is lowered elementwise (see
        # `translate_as_fieldop`), or for tuple-output scans whose stencil
        # body contains `tuple_get` chains on the carry parameter.
        if cpm.is_call_to(node, "tuple_get"):
            return self._visit_tuple_get(node, ctx=ctx)

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

    def _visit_tuple_get(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Generate the code for a possibly-nested ``tuple_get`` access.

        Descends along chains of tuple accesses (``x[i][j]...``), following
        let-bound aliases (``args_map``), until the tuple producer is found
        and the accessed element is visited.  Intermediate producers handled:

        - ``make_tuple`` — select the element;
        - applied lambdas (from e.g. CSE-substituted sub-tuples) — inline
          the lambda via ``_bind_inlined_args`` and descend into its body;
        - ``if_`` over tuples — distribute the remaining accesses over both
          branches.

        The access indices are collected innermost-last and consumed from the
        inside out.
        """
        assert cpm.is_call_to(node, "tuple_get")
        indices: list[int] = []
        expr: gtir.Expr = node
        current_ctx = ctx
        while True:
            if isinstance(expr, gtir.SymRef):
                resolved = self._resolve_args_map_symbol(expr, current_ctx)
                if resolved is not expr and not (
                    isinstance(resolved, gtir.SymRef) and str(resolved.id) == str(expr.id)
                ):
                    expr = resolved
                    continue
                if isinstance(expr.type, ts.TupleType):
                    # A tuple-typed symbol that is not let-bound (e.g. a
                    # program parameter of tuple-of-scalars type): expand it
                    # into a nested ``make_tuple`` of its flattened element
                    # symbols (``a`` → ``(a_0, (a_1_0, a_1_1))``), which are
                    # the SDFG symbols/containers the symbol was flattened
                    # into at the program level.
                    expr = _sym_tree_to_make_tuple(make_symbol_tree(str(expr.id), expr.type))
                    continue
            if cpm.is_call_to(expr, "tuple_get"):
                index_arg, inner = expr.args
                assert isinstance(index_arg, gtir.Literal)
                indices.append(int(index_arg.value))
                expr = inner
                continue
            if not indices:
                # Not a tuple access after all — visit the resolved node.
                return self.visit(expr, ctx=current_ctx)
            if cpm.is_call_to(expr, "make_tuple"):
                index = indices.pop()  # consume the innermost index first
                expr = expr.args[index]
                continue
            if isinstance(expr, gtir.FunCall) and isinstance(expr.fun, gtir.Lambda):
                # Applied lambda producing a tuple — inline it (binding its
                # arguments) and descend into its body; all remaining
                # indices apply to the tuple the lambda evaluates to.
                new_bindings = {
                    p.id: arg for p, arg in zip(expr.fun.params, expr.args, strict=True)
                }
                current_ctx = self._bind_inlined_args(new_bindings, current_ctx)
                expr = expr.fun.expr
                continue
            if cpm.is_call_to(expr, "if_"):
                # Distribute the tuple access over the branches, e.g.
                # `(cond ? {a, b} : {c, d})[i]` → `cond ? a[i]/...` —
                # needed for tuple-output scans whose stencil body has an
                # `if_` over tuple values.
                index = indices.pop()
                cond = self.visit(expr.args[0], ctx=current_ctx)
                true_val = self._visit_tuple_get(
                    self._fold_tuple_get(indices, index, expr.args[1]), ctx=current_ctx
                )
                false_val = self._visit_tuple_get(
                    self._fold_tuple_get(indices, index, expr.args[2]), ctx=current_ctx
                )
                code = format_builtin("if_", cond, true_val, false_val)
                return f"({code})"
            raise NotImplementedError(
                f"tuple_get of a non-literal tuple in stencil code is not supported: {node}"
            )

    @staticmethod
    def _fold_tuple_get(indices: list[int], outer_index: int, expr: gtir.Expr) -> gtir.FunCall:
        """Rebuild a ``tuple_get`` node from an index stack and consumed index.

        ``_fold_tuple_get([i0, i1], i2, expr)`` returns
        ``tuple_get(i0, tuple_get(i1, tuple_get(i2, expr)))``.
        """
        node: gtir.Expr = im.tuple_get(outer_index, expr)
        for i in reversed(indices):
            node = im.tuple_get(i, node)
        assert isinstance(node, gtir.FunCall)
        return node

    def _resolve_data_arg(self, node: gtir.SymRef, ctx: CodegenContext) -> DataRef | None:
        """Resolve a SymRef (possibly through args_map) to its DataRef.

        The symbol is first looked up directly in ``data_args`` (the common
        case: it is a field-operator lambda parameter).  If not found, the
        ``args_map`` chain — updated when let-lambdas are encountered in
        ``visit_FunCall`` — is followed until a symbol in ``data_args`` is
        reached.

        Returns ``None`` if the symbol does not resolve to a field argument
        (e.g. it refers to an SDFG data container or a symbol); callers fall
        back to ``visit_SymRef`` in that case.
        """
        symbol = str(node.id)
        if symbol in ctx.data_args:
            return ctx.data_args[symbol]

        seen: set[str] = {symbol}
        while symbol in ctx.args_map:
            mapped = ctx.args_map[symbol]
            if isinstance(mapped, gtir.SymRef):
                symbol = str(mapped.id)
                if symbol in ctx.data_args:
                    return ctx.data_args[symbol]
                if symbol in seen:
                    break
                seen.add(symbol)
            else:
                break

        # Fall back to root-level field containers: a field captured by
        # reference (a program argument used inside a field operator without
        # being an explicit lambda parameter, as e.g. produced by the
        # inlining passes of `apply_common_transforms`) has no entry in
        # ``data_args``, but its container is declared at the SDFG root;
        # fabricate the ``DataRef`` for it (mirroring ``make_field``).
        # The symbol can be iterator-typed (lambda parameter); reconstruct
        # the field type from the iterator's domain and element type.
        if symbol in ctx.root_symbols:
            field_type: ts.FieldType | None = None
            if isinstance(node.type, ts.FieldType):
                field_type = node.type
            elif isinstance(node.type, itir_ts.IteratorType) and node.type.position_dims == "unknown":
                field_type = ts.FieldType(
                    dims=node.type.defined_dims, dtype=node.type.element_type
                )
            if field_type is not None and isinstance(field_type.dtype, ts.ScalarType):
                if any(dim.kind == gtx_common.DimensionKind.LOCAL for dim in field_type.dims):
                    return None
                origin = tuple(
                    gtx_dace_args.range_start_symbol(symbol, dim) for dim in field_type.dims
                )
                return DataRef(symbol, field_type, origin)

        return None

    def _resolve_lifted_lambda(self, applied_lift: gtir.FunCall, ctx: CodegenContext) -> gtir.Lambda:
        """Return the lambda of an applied lift ``(↑f)(args...)``.

        ``apply_common_transforms`` (common subexpression elimination) may
        extract the lifted stencil into a let-bound symbol, so the argument
        of ``lift`` can arrive as a ``SymRef``; resolve it through
        ``args_map`` (like ``_resolve_args_map_symbol`` for call arguments).
        """
        assert isinstance(applied_lift.fun, gtir.FunCall)
        lifted_lambda = self._resolve_args_map_symbol(applied_lift.fun.args[0], ctx)
        assert isinstance(lifted_lambda, gtir.Lambda)
        return lifted_lambda

    def _resolve_args_map_symbol(self, arg: gtir.Expr, ctx: CodegenContext) -> gtir.Expr:
        """Resolve a symbol bound by an (inlined) let-lambda to its expression.

        Fused field operators pass shifted iterators and other expressions
        as lambda arguments (e.g. ``_cs_1`` -> ``⟪IDim→IDim, 1ₒ⟫(in_field)``).
        Symbols present in ``data_args`` are lambda parameters with an
        already-lowered field handle and are never resolved away; the
        identity check stops self-referential mappings
        (e.g. ``args_map={'in_field': SymRef('in_field')}``).
        """
        while isinstance(arg, gtir.SymRef):
            symbol = str(arg.id)
            if symbol in ctx.data_args or symbol not in ctx.args_map:
                break
            mapped = ctx.args_map[symbol]
            if mapped == arg or not isinstance(mapped, gtir.Expr):
                break
            arg = mapped
        return arg

    def _resolve_shifts(
        self, arg: gtir.Expr, ctx: CodegenContext
    ) -> tuple[gtir.SymRef, DataRef | None, dict[gtx_common.Dimension, tuple[str, bool]], set[str]]:
        """Walk a shift chain and return the field data and indices.

        Shared logic between ``_visit_deref`` and ``_single_neighbor_access``:
        collects cartesian and unstructured shifts, applies them
        (innermost first), and returns the resolved ``SymRef``,
        ``DataRef``, and index expressions.

        Returns:
            ``(sym_ref, data, indices, used_connectivities)`` where
            ``indices`` maps each dimension to ``(index_expr, is_dynamic)``
            and ``used_connectivities`` is the set of connectivity-table
            names referenced by unstructured shifts.
        """
        # List of (old_dim, replacement) pairs: multiple shifts on the same
        # dimension may be composed (e.g. nested field operators fused by
        # `FuseAsFieldOp` pass shifted iterators to inner lambdas, so a body
        # can contain `·⟪IDim, 1ₒ⟫(⟪IDim, 1ₒ⟫(in_field))`-shaped accesses).
        shift_replacements: list[tuple[gtx_common.Dimension, ShiftReplacement]] = []
        used_connectivities_in_shift: set[str] = set()
        while True:
            arg = self._resolve_args_map_symbol(arg, ctx)
            if not cpm.is_applied_shift(arg):
                break
            shift_call = arg.fun  # FunCall(SymRef("shift"), [offset_args...])
            shift_args = shift_call.args
            assert len(shift_args) >= 2 and len(shift_args) % 2 == 0
            for i in range(0, len(shift_args), 2):
                offset = shift_args[i]
                offset_value = shift_args[i + 1]
                if isinstance(offset, gtir.CartesianOffset):
                    old_dim = itir_misc.dim_from_axis_literal(offset.domain)
                    new_dim = itir_misc.dim_from_axis_literal(offset.codomain)
                    offset_str = self.visit(offset_value, ctx=ctx.child(is_inner_expr=True))
                    is_symbolic = isinstance(offset_value, (gtir.OffsetLiteral, gtir.Literal))
                    shift_replacements.append(
                        (
                            old_dim,
                            CartesianShiftReplacement(
                                new_dim=new_dim, offset_str=offset_str, is_symbolic=is_symbolic
                            ),
                        )
                    )
                else:
                    assert isinstance(offset, gtir.OffsetLiteral)
                    assert isinstance(offset.value, str)
                    offset_name = offset.value
                    offset_type = ctx.offset_provider_type[offset_name]
                    assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
                    conn_name = gtx_dace_args.connectivity_identifier(offset_name)
                    used_connectivities_in_shift.add(conn_name)
                    if isinstance(offset_value, gtir.OffsetLiteral):
                        shift_replacements.append(
                            (
                                offset_type.source_dim,
                                UnstructuredShiftReplacement(
                                    new_dim=offset_type.codomain,
                                    conn_name=conn_name,
                                    offset_str=str(offset_value.value),
                                    is_symbolic=False,
                                    max_neighbors=offset_type.max_neighbors,
                                ),
                            )
                        )
                    else:
                        offset_str = self.visit(offset_value, ctx=ctx.child(is_inner_expr=True))
                        shift_replacements.append(
                            (
                                offset_type.source_dim,
                                UnstructuredShiftReplacement(
                                    new_dim=offset_type.codomain,
                                    conn_name=conn_name,
                                    offset_str=offset_str,
                                    is_symbolic=False,
                                    max_neighbors=offset_type.max_neighbors,
                                ),
                            )
                        )
            arg = arg.args[0]  # the iterator

        if not isinstance(arg, gtir.SymRef):
            if cpm.is_applied_lift(arg):
                raise NotImplementedError(
                    "Nested reduction (reduce of an expression containing a lifted "
                    f"reduce operator) is not supported: {arg}"
                )
            raise ValueError(f"Unexpected deref argument: {arg}")
        data = self._resolve_data_arg(arg, ctx)

        if data is None:
            return arg, None, {}, used_connectivities_in_shift

        # Build the index expressions starting from the map variables
        # (field-operator domain dimensions), then apply shift replacements
        # to map them to the field's dimensions.  The replacements are
        # collected from the outermost shift inwards; this collection order
        # may consume the current map dimension at a late hop (e.g. for
        # ``⟪E2V, 0⟫(⟪C2E, 0⟫(it))`` evaluated over cells, the first hop is
        # the C2E replacement collected second), so the replacements are
        # applied in an order driven by which source dimension is currently
        # indexed.
        indices: dict[gtx_common.Dimension, tuple[str, bool]] = {
            dim: (map_var, dim in ctx.dynamic_map_indices)
            for dim, map_var in ctx.map_indices.items()
        }
        pending_shift_replacements = list(shift_replacements)
        while pending_shift_replacements:
            next_i = next(
                (
                    i
                    for i, (old_dim, _) in enumerate(pending_shift_replacements)
                    if old_dim in indices
                ),
                None,
            )
            if next_i is None:
                raise ValueError(
                    "Unable to compose shifts from the map variables: no pending"
                    " shift consumers an indexed dimension; missing sources:"
                    f" {[dim.value for dim, _ in pending_shift_replacements]};"
                    f" indexed dimensions: {[dim.value for dim in indices]}."
                )
            old_dim, repl = pending_shift_replacements.pop(next_i)
            old_index, _ = indices[old_dim]
            new_index = repl.apply(old_index)
            del indices[old_dim]
            indices[repl.new_dim] = (new_index, not repl.is_symbolic)

        return arg, data, indices, used_connectivities_in_shift

    def _visit_can_deref(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``can_deref(it)`` to a dereferencability predicate.

        Walks the shift chain applied to the iterator (as in
        ``_resolve_shifts``) and builds a conjunction of predicates, one
        per unstructured shift: a shift through the ``X`` connectivity by
        ``off`` is dereferencable iff ``gt_conn_X[{source_index}, {off}]``
        is not the skip value.  Cartesian shifts are always dereferencable
        inside the field-operator domain (the domain is shrunk by domain
        inference), so they only adjust the position used by the shifts
        further down the chain.

        The predicates are ordered innermost shift first, so that the
        short-circuiting ``&&`` does not evaluate a connectivity-table
        lookup whose row index is itself a skip value.
        """
        assert cpm.is_call_to(node, "can_deref") and len(node.args) == 1
        arg = node.args[0]

        # Position per dimension, starting at the field-operator map variables.
        indices = {dim: map_var for dim, map_var in ctx.map_indices.items()}
        predicates: list[str] = []
        while True:
            arg = self._resolve_args_map_symbol(arg, ctx)
            if not cpm.is_applied_shift(arg):
                break
            shift_args = arg.fun.args  # FunCall(SymRef("shift"), [offset_args...])
            assert len(shift_args) >= 2 and len(shift_args) % 2 == 0
            for i in range(0, len(shift_args), 2):
                offset = shift_args[i]
                offset_value = shift_args[i + 1]
                if isinstance(offset, gtir.CartesianOffset):
                    old_dim = itir_misc.dim_from_axis_literal(offset.domain)
                    new_dim = itir_misc.dim_from_axis_literal(offset.codomain)
                    if old_dim not in indices:
                        raise ValueError(f"No map variable for shifted dimension {old_dim}.")
                    old_index = indices.pop(old_dim)
                    if isinstance(offset_value, gtir.OffsetLiteral):
                        offset_str = str(offset_value.value)
                    else:
                        offset_str = self.visit(offset_value, ctx=ctx.child(is_inner_expr=True))
                    indices[new_dim] = (
                        old_index if offset_str == "0" else f"({old_index} + {offset_str})"
                    )
                else:
                    assert isinstance(offset, gtir.OffsetLiteral)
                    assert isinstance(offset.value, str)
                    offset_type = ctx.offset_provider_type[offset.value]
                    assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
                    conn_name = gtx_dace_args.connectivity_identifier(offset.value)
                    ctx.used_connectivities.add(conn_name)
                    if offset_type.source_dim not in indices:
                        raise ValueError(
                            f"No map variable for shifted dimension {offset_type.source_dim}."
                        )
                    old_index = indices.pop(offset_type.source_dim)
                    if isinstance(offset_value, gtir.OffsetLiteral):
                        offset_str = str(offset_value.value)
                    else:
                        offset_str = self.visit(offset_value, ctx=ctx.child(is_inner_expr=True))
                    neighbor_index = connectivity_table_index(
                        offset_type.max_neighbors, conn_name, old_index, offset_str
                    )
                    if offset_type.has_skip_values:
                        predicates.append(f"{neighbor_index} != {gtx_common._DEFAULT_SKIP_VALUE}")
                    indices[offset_type.codomain] = neighbor_index
            arg = arg.args[0]  # the iterator

        if not predicates:
            return "True"
        return "(" + " && ".join(reversed(predicates)) + ")"

    def _visit_deref(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``deref(expr)`` to a scalar or 1D field access.

        This mirrors ``_visit_deref``, ``_make_cartesian_shift`` and
        ``_make_unstructured_shift`` in ``lowering.gtir_to_sdfg_iterator``:
        the ``indices`` dict starts from the map variables (field-operator
        domain dimensions) and each shift — cartesian or unstructured —
        replaces ``old_dim`` (offset.domain / source_dim) with ``new_dim``
        (offset.codomain), adjusting the index by the offset or by a
        connectivity-table lookup.

        A dimension is *symbolic* when both the iterator index and the
        shift offset are compile-time known (``gtir.OffsetLiteral`` or
        ``gtir.Literal`` for cartesian; ``gtir.OffsetLiteral`` with a
        literal integer value for unstructured); a *dynamic* dimension is
        one whose offset is a runtime value.

        Symbolic dimensions are indexed on the input edge to the tasklet
        (memlet subset), exactly as in the SDFG lowering when all indices
        are ``SymbolExpr``.  Dynamic dimensions are indexed inside the
        tasklet.  Because DaCe's Python-to-C++ transpiler cannot lower
        multi-dimensional array access (``a[i, j, k]``), the memlet subset
        encodes all symbolic dimensions as exact indices and leaves the
        dynamic dimensions as full ranges; the resulting sub-array (with
        only the dynamic dimensions remaining) is passed to the tasklet
        and indexed with at most one index (``a[i]``).
        """
        deref_arg = self._resolve_args_map_symbol(node.args[0], ctx)
        if isinstance(deref_arg, gtir.Literal):
            # ``deref`` of a literal is the literal itself; appears e.g. in
            # ``make_const_list(deref(2))`` after the literal is passed as
            # a field-operator argument.
            return self.visit(deref_arg, ctx=ctx)
        if cpm.is_applied_lift(deref_arg):
            # `deref(↑f(x))` == `f(x)` — the applied-lift branch of
            # `visit_FunCall` evaluates the lifted lambda body at the
            # current location.
            return self.visit(deref_arg, ctx=ctx)

        _sym_ref, data, indices, used_connectivities_in_shift = self._resolve_shifts(
            node.args[0], ctx
        )

        if data is None:
            # Not a field argument — visit as a regular expression
            return self.visit(node.args[0], ctx=ctx)

        if isinstance(data.gt_type, ts.ScalarType):
            # Scalar access (no index).  NOTE: use the connector-safe name
            # (`_access_name`) — referencing the raw container name breaks
            # if DaCe transformations (scalar promotion/cloning) rename the
            # container, while keeping a tasklet connector alias intact.
            return self._access_name(data, ctx)

        assert isinstance(data.gt_type, ts.FieldType)

        # Record used connectivities so the caller can add them as inputs.
        ctx.used_connectivities.update(used_connectivities_in_shift)

        # When visiting an inner expression (e.g. a shift offset value
        # that is itself a ``deref``), use the old approach (inline
        # array access) instead of creating a ``field_inputs`` entry.
        # This avoids pointer-indexing issues when the inner deref
        # result is used as a dynamic index.
        if ctx.is_inner_expr:
            squeezed = ctx.scalar_field_dims.get(data.name, frozenset())
            parts = []
            for dim, origin in zip(data.gt_type.dims, data.origin, strict=True):
                if dim not in indices:
                    raise ValueError(f"No map variable for dimension {dim}.")
                if dim in squeezed:
                    continue
                index_expr, _ = indices[dim]
                parts.append(f"({index_expr} - ({origin}))")
            return f"{self._access_name(data, ctx)}[{', '.join(parts)}]"

        # Fields with a local dimension still need the full array inside
        # the tasklet because the local dimension is indexed by a loop
        # variable.  This is the same limitation as in the SDFG lowering,
        # where ``get_memlet_subset`` only handles ``SymbolExpr`` indices.
        # TODO(edopao): handle local dimensions via input edge too.
        if isinstance(data.gt_type.dtype, ts.ListType):
            squeezed = ctx.scalar_field_dims.get(data.name, frozenset())
            parts = []
            for dim, origin in zip(data.gt_type.dims, data.origin, strict=True):
                if dim not in indices:
                    raise ValueError(f"No map variable for dimension {dim}.")
                if dim in squeezed:
                    continue
                index_expr, _ = indices[dim]
                parts.append(f"({index_expr} - ({origin}))")
            return f"{self._access_name(data, ctx)}[{', '.join(parts)}]"

        # Build per-dimension index expressions for the field's dimensions
        # and classify each as symbolic (compile-time known) or dynamic
        # (runtime value).
        dim_info: list[tuple[gtx_common.Dimension, str, bool]] = []
        for dim, origin in zip(data.gt_type.dims, data.origin, strict=True):
            if dim not in indices:
                raise ValueError(f"No map variable for dimension {dim}.")
            index_expr, is_dynamic = indices[dim]
            index_expr = f"({index_expr} - ({origin}))"
            dim_info.append((dim, index_expr, is_dynamic))

        if not any(is_dynamic for _, _, is_dynamic in dim_info):
            # All indices are symbolic: field access on the input edge.
            # Register a scalar connector whose memlet reads one element
            # from the field at the computed multi-dimensional index.
            connector = ctx.fresh_deref_connector()
            # A 0-dimensional field has an empty subset list; its single
            # element is addressed with `0`.
            subset = ", ".join(index_expr for _, index_expr, _ in dim_info) or "0"
            ctx.field_inputs[connector] = (data.name, subset)
            return connector

        # Some indices are dynamic.  As in the SDFG lowering
        #  (``_visit_deref`` in ``gtir_to_sdfg_iterator.py``), the full
        #  array is passed to the tasklet and the dynamic indices are
        #  computed inline.  The multi-dimensional access is emitted on the
        #  connector-safe name (``_access_name``): ``add_tasklet`` renames
        #  the field operator argument to the (prefixed) tasklet connector,
        #  which makes DaCe's Python-to-C++ transpiler linearize the
        #  access correctly using the array strides.  Referencing the raw
        #  container name instead would either leave a dangling,
        #  non-transpilable multi-dimensional subscript on a global array
        #  pointer or, when the argument name equals the container name,
        #  produce a subscript whose dimensionality does not match the
        #  full-array memlet on the connector.  Scalar (size-one) container
        #  dimensions are squeezed out of the subscript, since DaCe requires
        #  the subscript dimensionality of the connector to match the number
        #  of non-scalar dimensions of the memlet subset.
        squeezed = ctx.scalar_field_dims.get(data.name, frozenset())
        return f"{self._access_name(data, ctx)}[{', '.join(index_expr for dim, index_expr, _ in dim_info if dim not in squeezed)}]"

    def _single_neighbor_access(
        self, node: gtir.FunCall, neighbor_var: str, ctx: CodegenContext
    ) -> ListElementAccess:
        """Build the scalar access expression for one neighbor.

        Args:
            node: A ``neighbors(offset, source)`` ``FunCall``.
            neighbor_var: The C variable used to index the neighbor
                (i.e. the connectivity table's second axis).
            ctx: The codegen context.

        Returns:
            A ``ListElementAccess`` with the access expression, the
            skip-value guard (or ``None``) and the dummy replacement value.
            The connectivity container name is also recorded in
            ``ctx.used_connectivities`` as a side effect.
        """
        assert len(node.args) == 2
        offset = node.args[0]
        source = node.args[1]

        assert isinstance(offset, gtir.OffsetLiteral)
        assert isinstance(offset.value, str), f"Expected string offset name, got '{offset.value}'"

        offset_type = ctx.offset_provider_type[offset.value]
        assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
        conn_name = gtx_dace_args.connectivity_identifier(offset.value)
        ctx.used_connectivities.add(conn_name)

        # Resolve any shift chain on the source (after FuseAsFieldOp,
        # the source of ``neighbors(...)`` can be a shifted expression,
        # e.g. ``shift(KDim, 1)(edge_f)``).  Same logic as
        # ``_visit_deref``: build an ``indices`` dict by applying shift
        # replacements, then replace the source dimension with the
        # connectivity-table index.
        _sym_ref, data, indices, used_conns = self._resolve_shifts(source, ctx)
        ctx.used_connectivities.update(used_conns)
        assert data is not None, f"Field argument '{source}' not found."
        assert isinstance(data.gt_type, ts.FieldType)

        source_dim = offset_type.source_dim
        source_index, _ = indices.get(source_dim, (None, None))
        assert source_index is not None, f"No map variable for source dimension {source_dim}."

        # Build the array access for one neighbor.  The field's dimension
        # that matches the connectivity's codomain is indexed through the
        # connectivity table; all other dimensions use their index from
        # ``indices`` (which may have been adjusted by cartesian or
        # unstructured shifts).
        neighbor_index = connectivity_table_index(
            offset_type.max_neighbors, conn_name, source_index, neighbor_var
        )
        squeezed = ctx.scalar_field_dims.get(data.name, frozenset())
        index_parts = []
        for dim, origin in zip(data.gt_type.dims, data.origin, strict=True):
            if dim in squeezed:
                continue
            if dim == offset_type.codomain:
                index_parts.append(f"({neighbor_index} - ({origin}))")
            else:
                idx, _ = indices.get(dim, (None, None))
                assert idx is not None, f"No map variable for dimension {dim}."
                index_parts.append(f"({idx}) - ({origin})")

        access_expr = f"{self._access_name(data, ctx)}[{', '.join(index_parts)}]"

        if not offset_type.has_skip_values:
            return ListElementAccess(expr=access_expr)

        mask = f"{neighbor_index} != {gtx_common._DEFAULT_SKIP_VALUE}"
        field_dtype = data.gt_type.dtype
        if isinstance(field_dtype, ts.ScalarType):
            dc_dtype = gtx_dace_args.as_dace_type(field_dtype)
        else:
            assert isinstance(field_dtype, ts.ListType)
            assert isinstance(field_dtype.element_type, ts.ScalarType)
            dc_dtype = gtx_dace_args.as_dace_type(field_dtype.element_type)
        # Use NaN for floating point, max value for integers
        if np.issubdtype(dc_dtype.as_numpy_dtype(), np.floating):
            dummy = "math.nan"
        else:
            dummy = str(dace.dtypes.max_value(dc_dtype))

        return ListElementAccess(expr=access_expr, mask=mask, dummy=dummy)

    def _single_lifted_map_element(
        self, node: gtir.FunCall, source: gtir.FunCall, neighbor_var: str, ctx: CodegenContext
    ) -> ListElementAccess:
        """Element access for ``neighbors(offset, (↑stencil)(args...))``.

        The source of the ``neighbors`` call is an applied lifted stencil —
        e.g. a field operator containing a nested ``reduce``, fused into the
        outer ``reduce`` by ``FuseAsFieldOp`` (see `test_nested_reduction` in
        `tests/next_tests/integration_tests/feature_tests/ffront_tests/test_reductions.py`).
        The n-th element of the neighbor list is the stencil body evaluated
        with the lifted iterators positioned at the n-th neighbor of the
        current location: the position along the connectivity's codomain
        dimension is the runtime connectivity-table entry
        ``conn[source_position, n]``.

        The body is visited in a child context whose ``map_indices`` entry
        for the codomain dimension holds that position expression, with the
        stencil parameters bound to the call arguments (any statements the
        body generates — e.g. the accumulation loop of the nested ``reduce``
        — are captured in a fresh ``pre_statements`` buffer and returned in
        ``ListElementAccess.body``; mutable state such as the temp counter,
        ``used_connectivities`` and ``field_inputs`` is shared with the
        parent context).  The caller must emit ``body`` at the
        element-evaluation site, guarded by ``mask`` when the connectivity
        has skip values.

        Args:
            node: The ``neighbors(offset, source)`` call.
            source: The applied lift ``(↑stencil)(args...)`` (already
                resolved through ``args_map``).
            neighbor_var: The C variable used to index the neighbor
                (i.e. the connectivity table's second axis).
            ctx: The codegen context.

        Returns:
            A ``ListElementAccess`` whose ``expr`` is the stencil body's
            result expression and whose non-empty ``body`` contains the
            statements computing it for the current neighbor.
        """
        offset = node.args[0]
        assert isinstance(offset, gtir.OffsetLiteral) and isinstance(offset.value, str)
        offset_type = ctx.offset_provider_type[offset.value]
        assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
        conn_name = gtx_dace_args.connectivity_identifier(offset.value)
        ctx.used_connectivities.add(conn_name)
        source_index = ctx.map_indices.get(offset_type.source_dim)
        assert source_index is not None, (
            f"No map variable for source dimension {offset_type.source_dim}."
        )
        neighbor_position = f"{conn_name}[{source_index}, {neighbor_var}]"

        lifted_lambda = self._resolve_lifted_lambda(source, ctx)
        new_bindings = {
            param.id: arg for param, arg in zip(lifted_lambda.params, source.args, strict=True)
        }
        child_ctx = self._bind_inlined_args(
            new_bindings,
            ctx.child(
                map_indices={**ctx.map_indices, offset_type.codomain: f"({neighbor_position})"},
                dynamic_map_indices=ctx.dynamic_map_indices | {offset_type.codomain},
                pre_statements=[],
            ),
        )
        element_expr = self.visit(lifted_lambda.expr, ctx=child_ctx)

        mask: str | None = None
        if offset_type.has_skip_values:
            # Skip invalid neighbors, matching `ReduceWithSkipValues` in the
            # SDFG lowering; the guard must also wrap the statements in
            # ``body`` since they read the connectivity table at the
            # neighbor position.
            mask = f"{neighbor_position} != {gtx_common._DEFAULT_SKIP_VALUE}"

        assert isinstance(node.type, ts.ListType)
        element_type = node.type.element_type
        assert isinstance(element_type, ts.ScalarType)
        dc_element_type = gtx_dace_args.as_dace_type(element_type)
        dummy = (
            "math.nan"
            if np.issubdtype(dc_element_type.as_numpy_dtype(), np.floating)
            else str(dace.dtypes.max_value(dc_element_type))
        )

        return ListElementAccess(
            expr=element_expr, mask=mask, dummy=dummy, body=tuple(child_ctx.pre_statements)
        )

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
        neighbors_offset = node.args[0]
        assert isinstance(neighbors_offset, gtir.OffsetLiteral) and isinstance(
            neighbors_offset.value, str
        )
        offset_type = ctx.offset_provider_type[neighbors_offset.value]
        assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
        max_neighbors = offset_type.max_neighbors

        neighbor_var = ctx.fresh_temp()
        result_var = ctx.fresh_temp()
        access, _ = self._single_map_element(node, neighbor_var, ctx)

        ctx.pre_statements.append(f"{result_var} = [0] * {max_neighbors}")
        ctx.pre_statements.append(f"for {neighbor_var} in range({max_neighbors}):")
        if access.mask is None:
            ctx.pre_statements.extend(f"    {line}" for line in access.body)
            ctx.pre_statements.append(f"    {result_var}[{neighbor_var}] = {access.expr}")
        elif not access.body:
            ctx.pre_statements.append(
                f"    {result_var}[{neighbor_var}] ="
                f" ({access.expr}) if {access.mask} else {access.dummy}"
            )
        else:
            # Guard the per-element statements on the connectivity-table
            # entry: the statements read the connectivity at the neighbor
            # position, which is out of bounds for a skipped neighbor.
            ctx.pre_statements.append(f"    if {access.mask}:")
            ctx.pre_statements.extend(f"        {line}" for line in access.body)
            ctx.pre_statements.append(f"        {result_var}[{neighbor_var}] = {access.expr}")
            ctx.pre_statements.append("    else:")
            ctx.pre_statements.append(f"        {result_var}[{neighbor_var}] = {access.dummy}")
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
        assert cpm.is_applied_reduce(node)
        assert len(node.fun.args) == 2
        assert len(node.args) >= 1

        op_expr = node.fun.args[0]
        if isinstance(op_expr, gtir.SymRef):
            op_name: str | None = str(op_expr.id)
            if len(node.args) != 1:
                raise NotImplementedError(
                    f"Reduce with builtin operation '{op_name}' supports a single "
                    f"argument list only, got {len(node.args)}."
                )
        elif isinstance(op_expr, gtir.Lambda):
            assert len(op_expr.params) == 1 + len(node.args)
            op_name = None
        else:
            raise NotImplementedError(f"Unexpected reduce operation: {op_expr}")
        init_val = self.visit(node.fun.args[1], ctx=ctx)
        # Cast the init value to the reduction dtype: a plain Python literal
        # (e.g. '0') would be transpiled by DaCe to a C++ 'int' accumulator,
        # truncating the floating-point element values at every update step.
        assert isinstance(node.type, ts.ScalarType)
        init_val = format_builtin(node.type.kind.name.lower(), init_val)

        result_var = ctx.fresh_temp()
        ctx.pre_statements.append(f"{result_var} = {init_val}")

        def _apply_reduce_op(args: list[str]) -> str:
            """Apply the reduce operation to the accumulator + element values."""
            if op_name is not None:
                return format_builtin(op_name, *args)
            assert isinstance(op_expr, gtir.Lambda)
            # Inline the fused op lambda with its parameters bound to the
            # already-lowered accumulator and per-list element accesses.
            new_string_bindings = {str(p.id): a for p, a in zip(op_expr.params, args, strict=True)}
            return self.visit(op_expr.expr, ctx=ctx.bind_strings(new_string_bindings))

        # --- Fuse reduce + elementwise list expression ---------------------
        # When any reduce argument list is a `neighbors` call, fuse the
        # reduce with the elementwise expression over the lists (this
        # happens when field operators over a neighbor dimension are fused
        # into the reduce, e.g. `neighbor_sum(a + a)`).  All `neighbors`
        # calls must be over the same connectivity; other arguments are
        # `make_const_list` broadcasts, local fields, or nested `map_list`
        # expressions (handled by `_single_map_element`).
        if any(cpm.is_call_to(arg, "neighbors") for arg in node.args):
            arguments = list(node.args)

            def _nbr_offset(arg: gtir.FunCall) -> str:
                offset_lit = arg.args[0]
                assert isinstance(offset_lit, gtir.OffsetLiteral) and isinstance(
                    offset_lit.value, str
                )
                return offset_lit.value

            nbr_offsets = {
                _nbr_offset(arg) for arg in arguments if cpm.is_call_to(arg, "neighbors")
            }
            if len(nbr_offsets) != 1:
                raise NotImplementedError(
                    "Reduce over neighbor lists of different offsets: " + str(list(nbr_offsets))
                )
            loop_var = ctx.fresh_temp()
            elements = [self._single_map_element(arg, loop_var, ctx) for arg in arguments]
            local_size: int | None = None
            for _element, sz in elements:
                if sz is not None:
                    if local_size is None:
                        local_size = sz
                    elif sz != local_size:
                        raise ValueError(
                            f"Mismatching sizes of the local dimensions in reduce node {node}."
                        )
            if local_size is None:
                raise ValueError(f"Missing information on local dimension for reduce in {node}.")
            accumulate = (
                f"{result_var} = {_apply_reduce_op([result_var, *[e.expr for e, _ in elements]])}"
            )
            masks = [e.mask for e, _ in elements if e.mask is not None]
            # Skip invalid neighbors: guard the accumulation (and the
            # per-element statements such as a nested reduce loop, which
            # reads the connectivity table at the neighbor position) on the
            # connectivity-table entries, matching `ReduceWithSkipValues`
            # in the SDFG lowering.
            guard = " and ".join(masks) if masks else None
            indent = "        " if guard is not None else "    "
            # Per-element statements (e.g. the accumulation loop of a nested
            # reduce, see `_single_lifted_map_element`) are emitted inside
            # the loop body, before the accumulation.
            body = [f"{indent}{line}" for e, _ in elements for line in e.body]
            ctx.pre_statements.append(f"for {loop_var} in range({local_size}):")
            if guard is not None:
                ctx.pre_statements.append(f"    if {guard}:")
            ctx.pre_statements.extend(body)
            ctx.pre_statements.append(f"{indent}{accumulate}")
            return result_var

        inner = node.args[0]
        if len(node.args) != 1:
            raise NotImplementedError(
                "Reduce zipping multiple non-`neighbors` lists is not supported."
            )

        # --- Fuse reduce + deref of local field ---------------------------
        # Under the fieldview pipeline (without `FuseAsFieldOp`), a
        # ``neighbor_sum(field(V2E), axis=V2EDim)`` lowers to
        # ``reduce(plus, 0)(deref(__arg0))`` where ``__arg0`` is a field
        # with a local dimension (e.g. ``Field[[Vertex, V2EDim], int32]``).
        # Fuse the reduce with the local field: iterate over the local
        # dimension and accumulate.
        if cpm.is_call_to(inner, "deref") or isinstance(inner, gtir.SymRef):
            inner_ref = inner.args[0] if cpm.is_call_to(inner, "deref") else inner
            if isinstance(inner_ref, gtir.SymRef):
                data = self._resolve_data_arg(inner_ref, ctx)
                if (
                    data is not None
                    and isinstance(data.gt_type, ts.FieldType)
                    and isinstance(data.gt_type.dtype, ts.ListType)
                ):
                    local_offset = data.gt_type.dtype.offset_type
                    assert local_offset is not None
                    offset_provider_t = ctx.offset_provider_type.get(local_offset.value)
                    if isinstance(offset_provider_t, gtx_common.NeighborConnectivityType):
                        local_size = offset_provider_t.max_neighbors
                        loop_var = ctx.fresh_temp()
                        element_access, _ = self._single_map_element(inner, loop_var, ctx)
                        mask = element_access.mask
                        if mask is None and offset_provider_t.has_skip_values:
                            # The local field was materialized with a dummy
                            # value for skipped neighbors (see the stree
                            # ``neighbors`` lowering); skipped elements must
                            # not contribute to the reduction.  The mask is
                            # reconstructed from the connectivity table, as in
                            # ``ReduceWithSkipValues`` in the SDFG lowering.
                            conn_name = gtx_dace_args.connectivity_identifier(local_offset.value)
                            ctx.used_connectivities.add(conn_name)
                            source_index = ctx.map_indices.get(offset_provider_t.source_dim)
                            assert source_index is not None, (
                                f"No map variable for source dimension "
                                f"{offset_provider_t.source_dim}."
                            )
                            mask = (
                                f"{conn_name}[{source_index}, {loop_var}]"
                                f" != {gtx_common._DEFAULT_SKIP_VALUE}"
                            )
                        accumulate = (
                            f"{result_var} = {_apply_reduce_op([result_var, element_access.expr])}"
                        )
                        indent = "        " if mask is not None else "    "
                        body = [f"{indent}{line}" for line in element_access.body]
                        ctx.pre_statements.append(f"for {loop_var} in range({local_size}):")
                        if mask is not None:
                            ctx.pre_statements.append(f"    if {mask}:")
                        ctx.pre_statements.extend(body)
                        ctx.pre_statements.append(f"{indent}{accumulate}")
                        return result_var

        # --- Fuse reduce + map_list ---------------------------------------
        if cpm.is_applied_map(inner):
            # Fuse reduce + map_list into a scalar accumulation loop
            # (init; for n in range(N): result = reduce_op(result,
            # map_op(elem_0, elem_1, ...))) to avoid Python list creation.
            map_op = inner.fun.args[0]
            map_args = inner.args

            # Determine local size from the first non-const argument.
            local_size = None
            for arg_node in map_args:
                _element, sz = self._single_map_element(arg_node, "__dummy", ctx)
                if sz is not None:
                    local_size = sz
                    break
            if local_size is None:
                raise ValueError(f"Missing information on local dimension for map_list in {node}.")

            loop_var = ctx.fresh_temp()
            # Compute the element expression for each map_list argument.
            elements = [self._single_map_element(arg, loop_var, ctx) for arg in map_args]
            mapped_expr = self._map_op_expr(map_op, [e.expr for e, _ in elements], ctx)

            # Guard the accumulation on the connectivity-table entries of
            # the neighbor arguments with skip values (matching
            # `ReduceWithSkipValues` in the SDFG lowering): this excludes a
            # skipped neighbor from the accumulation altogether, also when
            # the map expression mixes neighbor values with constants or
            # scalars (e.g. `where(mask, inp(V2E), 1)`).
            masks = [element.mask for element, _ in elements if element.mask is not None]
            accumulate = f"{result_var} = {_apply_reduce_op([result_var, mapped_expr])}"
            if masks:
                guard = " and ".join(masks)
                # Per-element statements (e.g. a nested reduce loop) go
                # inside the skip-value guard: they read the connectivity
                # table at the neighbor position, which is out of bounds
                # for a skipped neighbor.
                body = [f"        {line}" for e, _ in elements for line in e.body]
                ctx.pre_statements.extend(
                    [
                        f"for {loop_var} in range({local_size}):",
                        f"    if {guard}:",
                        *body,
                        f"        {accumulate}",
                    ]
                )
            else:
                body = [f"    {line}" for e, _ in elements for line in e.body]
                ctx.pre_statements.extend(
                    [
                        f"for {loop_var} in range({local_size}):",
                        *body,
                        f"    {accumulate}",
                    ]
                )

        else:
            loop_var = ctx.fresh_temp()
            reduce_expr = self.visit(inner, ctx=ctx)
            ctx.pre_statements.extend(
                [
                    f"for {loop_var} in {reduce_expr}:",
                    f"    {result_var} = {_apply_reduce_op([result_var, loop_var])}",
                ]
            )

        return result_var

    def _visit_map_list(self, node: gtir.FunCall, *, ctx: CodegenContext) -> str:
        """Lower ``map_list(op)(args...)`` to a Python list comprehension.

        Generates:
        ``[{builtin}(__arg0, __arg1, ...) for __n in range({local_size})]``
        """
        assert cpm.is_applied_map(node)

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

    def _access_name(self, data: DataRef, ctx: CodegenContext) -> str:
        """Name used to reference a field inside tasklet code.

        The lambda parameter name is preferred when the field is a field
        operator argument: ``add_tasklet`` renames it to the (prefixed)
        tasklet connector, which makes DaCe's Python-to-C++ transpiler
        linearize multi-dimensional access correctly (using the array
        stride symbols).  Raw container names are only safe for
        one-dimensional access to non-transient SDFG arguments.
        """
        for param_name, other in ctx.data_args.items():
            if other is data:
                return param_name
        return data.name

    def _map_op_expr(self, map_op: gtir.Expr, element_exprs: list[str], ctx: CodegenContext) -> str:
        """Build the expression ``map_op(elem_0, elem_1, ...)``.

        For a lambda ``map_op``, binds the parameter names to the element
        expressions (via ``string_args``) and visits the body.
        """
        if isinstance(map_op, gtir.SymRef):
            return format_builtin(str(map_op.id), *element_exprs)
        if isinstance(map_op, gtir.Lambda):
            lambda_ctx = ctx.bind_strings(
                {str(p.id): e for p, e in zip(map_op.params, element_exprs, strict=True)}
            )
            return self.visit(map_op.expr, ctx=lambda_ctx)
        raise NotImplementedError(f"Non-SymRef/Lambda map_list operation not supported: {map_op}")

    def _single_map_element(
        self, arg: gtir.Node, loop_var: str, ctx: CodegenContext
    ) -> tuple[ListElementAccess, int | None]:
        """Generate the Python expression for one element of a ``map_list`` argument.

        Args:
            arg: An argument to ``map_list`` — typically ``neighbors(...)``,
                ``make_const_list(...)``, or a ``SymRef`` / ``deref`` of a
                local field.
            loop_var: The C variable used to index the local (neighbor)
                dimension.
            ctx: The codegen context.

        Returns:
            A ``(element, max_neighbors)`` tuple where ``element`` is the
            ``ListElementAccess`` for one element (with skip-value handling
            info) and ``max_neighbors`` is the size of the local dimension
            (or ``None`` for const lists).
        """
        # Case 1: neighbors(offset, source) — single neighbor access
        if cpm.is_call_to(arg, "neighbors"):
            source = self._resolve_args_map_symbol(arg.args[1], ctx)
            if cpm.is_applied_lift(source):
                element = self._single_lifted_map_element(arg, source, loop_var, ctx)
            else:
                element = self._single_neighbor_access(arg, loop_var, ctx)
            offset_lit = arg.args[0]
            assert isinstance(offset_lit, gtir.OffsetLiteral) and isinstance(offset_lit.value, str)
            offset_type = ctx.offset_provider_type[offset_lit.value]
            assert isinstance(offset_type, gtx_common.NeighborConnectivityType)
            return element, offset_type.max_neighbors

        # Case 2: make_const_list(value) — broadcast, no indexing
        if cpm.is_call_to(arg, "make_const_list"):
            assert len(arg.args) == 1
            return ListElementAccess(expr=self.visit(arg.args[0], ctx=ctx)), None

        # Case 3: nested map_list(op)(args...) — fuse the inner map at
        # element level (e.g. from `where(mask, inp(V2E), 1)` or common
        # subexpressions after FuseAsFieldOp).  Skip-value masks of the
        # inner elements are combined: the position is only valid if all
        # inner elements are valid (combine masks with ``and``).
        if cpm.is_applied_map(arg):
            inner_op = arg.fun.args[0]
            inner_elements = [self._single_map_element(a, loop_var, ctx) for a in arg.args]
            expr = self._map_op_expr(inner_op, [e.expr for e, _ in inner_elements], ctx)
            masks = [e.mask for e, _ in inner_elements if e.mask is not None]
            mask = " and ".join(masks) if masks else None
            inner_size = next((sz for _, sz in inner_elements if sz is not None), None)
            return ListElementAccess(expr=expr, mask=mask), inner_size

        # Case 3b: if_(cond, true_list, false_list) — evaluate the branches
        # elementwise; the condition is a scalar expression.
        if cpm.is_call_to(arg, "if_"):
            cond = self.visit(arg.args[0], ctx=ctx)
            true_element, true_size = self._single_map_element(arg.args[1], loop_var, ctx)
            false_element, false_size = self._single_map_element(arg.args[2], loop_var, ctx)
            masks = [e.mask for e in (true_element, false_element) if e.mask is not None]
            mask = " and ".join(masks) if masks else None
            dummy = true_element.dummy or false_element.dummy
            expr = f"({true_element.expr}) if {cond} else ({false_element.expr})"
            return (
                ListElementAccess(expr=expr, mask=mask, dummy=dummy),
                true_size if true_size is not None else false_size,
            )

        # Case 4: deref(SymRef) or SymRef to a local field — index by
        # map vars + loop var.  FuseAsFieldOp wraps local fields in
        # ``deref`` before passing them to ``map_list``.
        symref = arg
        if cpm.is_call_to(arg, "deref") and isinstance(arg.args[0], gtir.SymRef):
            symref = arg.args[0]
        if isinstance(symref, gtir.SymRef):
            data = self._resolve_data_arg(symref, ctx)
            if data is not None and isinstance(data.gt_type, ts.FieldType):
                # Local data — an extra (local) dimension indexed by the
                # loop variable.  Two shapes:
                # - field `dims` (all covered by map dims) + `ListType`
                #   dtype (internal local field), or
                # - a field with one extra dimension not covered by the map
                #   domain, e.g. an external local field
                #   `Field[[Vertex, V2EDim]]` used as a neighbor list.
                indices: list[str] = []
                unmapped_dims: list[gtx_common.Dimension] = []
                for dim, origin in zip(data.gt_type.dims, data.origin, strict=True):
                    map_var = ctx.map_indices.get(dim)
                    if map_var is None:
                        unmapped_dims.append(dim)
                    else:
                        indices.append(f"({map_var}) - ({origin})")

                local_offset_dim = None
                if isinstance(data.gt_type.dtype, ts.ListType):
                    list_dtype = data.gt_type.dtype
                    if list_dtype.offset_type is None or list_dtype.offset_type == _CONST_DIM:
                        # Broadcast (const) list: all list elements are
                        # equal, so the field is stored without the local
                        # dimension and the same scalar value is read at
                        # every list position.
                        connector = ctx.fresh_deref_connector()
                        ctx.field_inputs[connector] = (
                            data.name,
                            ",".join(indices) if indices else "0",
                        )
                        return ListElementAccess(expr=connector), None
                    assert not unmapped_dims, f"No map variable for dimensions {unmapped_dims}."
                    local_offset_dim = list_dtype.offset_type
                elif len(unmapped_dims) == 1:
                    local_offset_dim = unmapped_dims[0]
                else:
                    # A scalar-dtype field fully covered by the map domain:
                    # no local dimension; visit it as a regular (scalar)
                    # expression.
                    return ListElementAccess(expr=self.visit(arg, ctx=ctx)), None

                if local_offset_dim is not None:
                    offset_provider_t = ctx.offset_provider_type.get(local_offset_dim.value)
                    if not isinstance(offset_provider_t, gtx_common.NeighborConnectivityType):
                        # The local dimension is not a known neighbor
                        # connectivity (e.g. an explicit local field dim not
                        # registered in the offset provider): its extent is
                        # unknown at compile time.  Fall back to a raw access
                        # (via the connector name, so DaCe linearizes the
                        # multi-dimensional subscript) and report the size as
                        # unknown — callers gather the list size from another
                        # list argument or fail with "Missing information on
                        # local dimension".
                        return (
                            ListElementAccess(
                                expr=f"{self._access_name(data, ctx)}"
                                f"[{', '.join([*indices, loop_var])}]"
                            ),
                            None,
                        )
                    max_neighbors = offset_provider_t.max_neighbors
                    mask = None
                    dummy = ""
                    if offset_provider_t.has_skip_values:
                        # Skipped neighbors hold the dummy value in the
                        # materialized field; consumers must exclude them.
                        # The mask is derived from the connectivity table,
                        # as in ``ReduceWithSkipValues`` / ``_visit_map``
                        # of the SDFG lowering.
                        conn_name = gtx_dace_args.connectivity_identifier(local_offset_dim.value)
                        ctx.used_connectivities.add(conn_name)
                        source_index = ctx.map_indices.get(offset_provider_t.source_dim)
                        assert source_index is not None, (
                            f"No map variable for source dimension {offset_provider_t.source_dim}."
                        )
                        mask = (
                            f"{conn_name}[{source_index}, {loop_var}]"
                            f" != {gtx_common._DEFAULT_SKIP_VALUE}"
                        )
                        field_type = data.gt_type
                        assert isinstance(field_type, ts.FieldType)
                        element_type: ts.DataType = (
                            field_type.dtype.element_type
                            if isinstance(field_type.dtype, ts.ListType)
                            else field_type.dtype
                        )
                        assert isinstance(element_type, ts.ScalarType)
                        dc_element_type = gtx_dace_args.as_dace_type(element_type)
                        dummy = (
                            "math.nan"
                            if np.issubdtype(dc_element_type.as_numpy_dtype(), np.floating)
                            else str(dace.dtypes.max_value(dc_element_type))
                        )
                    # Pass the field through a 1-D slice connector on the
                    # input edge: the memlet subset fixes the global
                    # dimensions at their exact indices and ranges over
                    # the full local dimension, so the tasklet code only
                    # needs a 1-D access.  DaCe's Python-to-C++ transpiler
                    # cannot lower multi-dimensional access to transient
                    # (temp) arrays in tasklet code.
                    connector = ctx.fresh_deref_connector()
                    all_dims = gtx_common.order_dimensions([*data.gt_type.dims, local_offset_dim])
                    subset_parts = list(indices)
                    subset_parts.insert(int(all_dims.index(local_offset_dim)), f"0:{max_neighbors}")
                    ctx.field_inputs[connector] = (data.name, ",".join(subset_parts))
                    return (
                        ListElementAccess(expr=f"{connector}[{loop_var}]", mask=mask, dummy=dummy),
                        max_neighbors,
                    )

            if data is not None and isinstance(data.gt_type, ts.ScalarType):
                # Scalar: no indexing — use the connector-safe name
                # (`_access_name`), see `_visit_deref`.
                return ListElementAccess(expr=self._access_name(data, ctx)), None

        # Case 5: generic — visit and index by loop variable
        return ListElementAccess(expr=f"{self.visit(arg, ctx=ctx)}[{loop_var}]"), None

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
        if cpm.is_applied_map(inner):
            map_op = inner.fun.args[0]
            map_args = inner.args

            # Evaluate each argument at the given index; neighbours read
            # through a connectivity with skip values fall back to the
            # dummy value (as if the full list had been materialized).
            indexed_args = []
            for arg in map_args:
                element, _size = self._single_map_element(arg, index, ctx)
                # Per-element statements (e.g. a nested reduce loop) are
                # evaluated once, preceding the expression they compute.
                # When masked, guard them — they read the connectivity
                # table at the neighbor position, which is out of bounds
                # for a skipped neighbor (and ``element.expr`` is only
                # evaluated when the mask holds, see the conditional below).
                if element.body and element.mask is not None:
                    ctx.pre_statements.append(f"if {element.mask}:")
                    ctx.pre_statements.extend(f"    {line}" for line in element.body)
                else:
                    ctx.pre_statements.extend(element.body)
                indexed_args.append(
                    f"({element.expr}) if {element.mask} else {element.dummy}"
                    if element.mask is not None
                    else element.expr
                )

            # Apply the map operation to the indexed arguments.
            return self._map_op_expr(map_op, indexed_args, ctx)

        # Fuse list_get + neighbors: single neighbor access at the index.
        if cpm.is_call_to(inner, "neighbors"):
            source = self._resolve_args_map_symbol(inner.args[1], ctx)
            if cpm.is_applied_lift(source):
                access = self._single_lifted_map_element(inner, source, index, ctx)
            else:
                access = self._single_neighbor_access(inner, index, ctx)
            if access.body and access.mask is not None:
                ctx.pre_statements.append(f"if {access.mask}:")
                ctx.pre_statements.extend(f"    {line}" for line in access.body)
            else:
                ctx.pre_statements.extend(access.body)
            return (
                f"({access.expr}) if {access.mask} else {access.dummy}"
                if access.mask is not None
                else access.expr
            )

        # `list_get` on an (external) local field — e.g. a field with an
        # extra local dimension like `Field[[Vertex, V2EDim]]` — index the
        # local dimension inside the tasklet (single subscript, chaining
        # subscripts would be rejected by DaCe's tasklet validation).
        inner_symref = inner.args[0] if cpm.is_call_to(inner, "deref") else inner
        if isinstance(inner_symref, gtir.SymRef):
            data = self._resolve_data_arg(inner_symref, ctx)
            if (
                data is not None
                and isinstance(data.gt_type, ts.FieldType)
                and (
                    isinstance(data.gt_type.dtype, ts.ListType)
                    or any(dim not in ctx.map_indices for dim in data.gt_type.dims)
                )
            ):
                element, _size = self._single_map_element(inner, index, ctx)
                return element.expr

        return f"{self.visit(inner, ctx=ctx)}[{index}]"


def generate_tasklet_code(
    expr: gtir.Expr,
    *,
    data_args: dict[str, DataRef],
    map_indices: dict[gtx_common.Dimension, str],
    offset_provider_type: gtx_common.OffsetProviderType,
    args_map: dict[str, gtir.Node] | None = None,
    string_args: dict[str, str] | None = None,
    temp_counter: itertools.count[int] | None = None,
    root_symbols: frozenset[str] = frozenset(),
    scalar_field_dims: dict[str, frozenset[gtx_common.Dimension]] | None = None,
) -> tuple[str, list[str], set[str], dict[str, tuple[str, str]]]:
    """Generate the Python code for a tasklet body.

    Args:
        expr: The GTIR expression (stencil body) to lower.
        data_args: Mapping from lambda parameter names to ``DataRef``.
        map_indices: Mapping from ``Dimension`` to map variable name.
        offset_provider_type: The offset provider type table.
        args_map: Initial args_map for let-lambda inlining (usually empty).
        string_args: Initial mapping of symbol names to code strings, resolved
            literally by the code generator (e.g. for a scan carry variable
            defined in a pre-statement).
        temp_counter: Optional shared counter for temporary/connector names.
            Callers generating the code for multiple expressions that end up
            in the same tasklet (e.g. the leaves of a tuple-output scan)
            must pass the same counter to all calls, so that generated
            connector names do not collide.
        root_symbols: Names of SDFG-level symbols and data containers, for
            the ``SymRef`` fallback in the code generator.

    Returns:
        A tuple ``(expr_code, pre_statements, used_connectivities,
        field_inputs)`` where ``expr_code`` is the final Python expression,
        ``pre_statements`` is a list of Python statements that must appear
        before the final expression (join them as: ``"\\n".join(pre_statements)
        + f"\\n__out = {expr_code}"``), ``used_connectivities`` is the set of
        connectivity-table names (offset names) referenced by ``neighbors`` in
        the stencil body, and ``field_inputs`` maps connector names to
        ``(field_name, subset)`` pairs for scalar field accesses that should
        be materialised on the input edge (memlet) rather than inside the
        tasklet code.
    """
    ctx = CodegenContext(
        args_map=args_map or {},
        string_args=string_args or {},
        data_args=data_args,
        map_indices=map_indices,
        offset_provider_type=offset_provider_type,
        root_symbols=root_symbols,
        scalar_field_dims=scalar_field_dims or {},
    )
    if temp_counter is not None:
        ctx = dataclasses.replace(ctx, _temp_counter=temp_counter)
    expr_code = StreePythonCodegen().visit(expr, ctx=ctx)
    return expr_code, ctx.pre_statements, ctx.used_connectivities, ctx.field_inputs


def generate_list_tasklet_code(
    expr: gtir.Expr,
    *,
    data_args: dict[str, DataRef],
    map_indices: dict[gtx_common.Dimension, str],
    offset_provider_type: gtx_common.OffsetProviderType,
    args_map: dict[str, gtir.Node] | None = None,
    local_index: str,
    root_symbols: frozenset[str] = frozenset(),
    scalar_field_dims: dict[str, frozenset[gtx_common.Dimension]] | None = None,
) -> tuple[str, list[str], set[str], dict[str, tuple[str, str]]]:
    """Generate the tasklet code for a single element of a list expression.

    Used to lower ``as_fieldop`` field operators computing a field with
    ``ListType`` dtype (e.g. produced by ``neighbors(...)`` in the stencil
    body): the tasklet is placed in a map scope that iterates over the
    field domain extended with the local dimension, and writes one element
    of the local dimension, indexed by ``local_index`` (the map variable
    declared by the caller for the local dimension).

    Skipped neighbors (connectivity tables with skip values) write the
    dummy value (``math.nan`` or the dtype's max value), matching the
    materialization in ``_visit_neighbors`` of the SDFG lowering.

    Args:
        expr: The GTIR expression (stencil body) to lower; must be typed
            ``ListType``.
        data_args: Mapping from lambda parameter names to ``DataRef``.
        map_indices: Mapping from ``Dimension`` to map variable name —
            only the *global* dimensions (the local dimension is given
            by ``local_index``).
        offset_provider_type: The offset provider type table.
        args_map: Initial args_map for let-lambda inlining (usually empty).
        local_index: Map variable (C variable name) indexing the local
            dimension.

    Returns:
        Same tuple as ``generate_tasklet_code``: ``(expr_code,
        pre_statements, used_connectivities, field_inputs)``.
    """
    assert isinstance(expr.type, ts.ListType)
    ctx = CodegenContext(
        args_map=args_map or {},
        data_args=data_args,
        map_indices=map_indices,
        offset_provider_type=offset_provider_type,
        root_symbols=root_symbols,
        scalar_field_dims=scalar_field_dims or {},
    )
    element, _local_size = StreePythonCodegen()._single_map_element(expr, local_index, ctx)
    # Per-element statements (e.g. a nested reduce loop) precede the
    # expression they compute; when masked, guard them — they read the
    # connectivity table at the neighbor position, which is out of bounds
    # for a skipped neighbor (and `element.expr` is only used when the mask
    # holds, see the conditional below).
    if element.body and element.mask is not None:
        ctx.pre_statements.append(f"if {element.mask}:")
        ctx.pre_statements.extend(f"    {line}" for line in element.body)
    else:
        ctx.pre_statements.extend(element.body)
    expr_code = element.expr
    if element.mask is not None:
        dummy = element.dummy
        if not dummy:
            # Skipped list positions are filled with a dummy value (nan for
            # floating point, the dtype's max value for integers), matching
            # ``_visit_map`` in the SDFG lowering.
            element_type = expr.type.element_type
            assert isinstance(element_type, ts.ScalarType)
            dc_element_type = gtx_dace_args.as_dace_type(element_type)
            dummy = (
                "math.nan"
                if np.issubdtype(dc_element_type.as_numpy_dtype(), np.floating)
                else str(dace.dtypes.max_value(dc_element_type))
            )
        expr_code = f"({expr_code}) if {element.mask} else {dummy}"
    return expr_code, ctx.pre_statements, ctx.used_connectivities, ctx.field_inputs
