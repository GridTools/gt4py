# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fuse coefficient producers into vertical scans before SDFG lowering.

Without fusion, a scan can read coefficient fields that a separate producer
has just written to memory. Moving eligible expressions into the scan lets it
compute coefficients at the level where they are consumed. For example, in
pseudocode where `b` is already an input field:

    # Before.
    p = producer(x)
    a = coefficient(p)
    y = scan(step, initial, a, b)

    # After.
    p = producer(x)
    carry = initial
    for k in vertical_order:
        carry = step(carry, coefficient_at_level(p, k), b[k])
        y[k] = carry

The pass considers all eligible immediate producers, not just one input.
Newly exposed producers such as `producer(x)` are not revisited. A producer
whose result is still needed elsewhere can remain outside the scan as well.
Fusion preserves the recurrence and does not guarantee faster execution.

The DaCe translator enables this pass with `fuse_scan_inputs=True` in
`auto_optimize_args`; it defaults to off and requires auto-optimization.
`scan_fusion_scope="immediate"` uses one producer layer after the standard
field-view transforms. The optional `"field_operator"` scope first calls
`normalize_scan_producers` while function boundaries are still visible.
That preparation combines local coefficient arithmetic but preserves calls
to upstream field operators. One prepared producer can therefore contain
several arithmetic operations without absorbing its caller's upstream work.

The translator owns the ordering: optional producer preparation, standard
field-view transforms, scan-input fusion, then field-view transforms again
if fusion rewrote the IR. The second normalization refreshes the domains of
new expressions before SDFG lowering. Normal SDFG optimization follows.

`fuse_scan_inputs` checks symbolic input/output overlap and uses the existing
field-operator fusion helper. `_LiftScanShifts` adapts supported vertical
reads to DaCe's scalar scan arguments; unsupported shifts keep the original
scan. Callers must not supply overlapping runtime buffers under different
field names, since the symbolic checks cannot detect those aliases.
"""

from typing import Any, TypeGuard

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import (
    fuse_as_fieldop,
    inline_lambdas,
    remap_symbols,
    symbol_ref_utils,
)
from gt4py.next.iterator.type_system import inference
from gt4py.next.type_system import type_specifications as ts


def _is_non_scan_fieldop(node: ir.Expr) -> TypeGuard[ir.FunCall]:
    return cpm.is_applied_as_fieldop(node) and not cpm.is_call_to(node.fun.args[0], "scan")


class _UnsupportedScanInput(Exception):
    """Keep the original statement when a fused scan needs unsupported accesses."""


class _LiftScanShifts(eve.NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.FunCall:
        if not (cpm.is_applied_as_fieldop(node) and cpm.is_call_to(node.fun.args[0], "scan")):
            return self.generic_visit(node, **kwargs)
        scan = node.fun.args[0]
        if not isinstance(scan.args[0], ir.Lambda) or len(node.fun.args) != 2:
            raise _UnsupportedScanInput()
        stencil, restore_scan = ir_misc.unwrap_scan(scan)
        domain = node.fun.args[1]
        arguments = dict(zip((str(p.id) for p in stencil.params), node.args, strict=True))
        # Do not lift accesses through a shadowed parameter or another scan.
        for expr in stencil.expr.pre_walk_values():
            if isinstance(expr, ir.Lambda) and any(str(p.id) in arguments for p in expr.params):
                raise _UnsupportedScanInput()
            if isinstance(expr, ir.FunCall) and cpm.is_call_to(expr, "scan"):
                raise _UnsupportedScanInput()
        used_names = {
            str(expr.id) for expr in node.pre_walk_values() if isinstance(expr, (ir.Sym, ir.SymRef))
        }
        added: dict[str, tuple[str, ir.Expr]] = {}

        class ReplaceShifts(eve.NodeTranslator):
            def visit_FunCall(self, expr: ir.FunCall, **kwargs: Any) -> ir.Expr:
                if not cpm.is_applied_shift(expr):
                    return self.generic_visit(expr, **kwargs)
                # Lift fixed vertical reads, including full/half-level conversions.
                if not isinstance(expr.args[0], ir.SymRef):
                    raise _UnsupportedScanInput()
                name = str(expr.args[0].id)
                offsets = expr.fun.args
                if name not in arguments or len(offsets) != 2:
                    raise _UnsupportedScanInput()
                axis, distance = offsets
                if not (
                    isinstance(axis, ir.CartesianOffset)
                    and axis.domain.kind == common.DimensionKind.VERTICAL
                    and common.as_non_staggered(
                        common.Dimension(axis.domain.value, axis.domain.kind)
                    )
                    == common.as_non_staggered(
                        common.Dimension(axis.codomain.value, axis.codomain.kind)
                    )
                    and isinstance(distance, ir.OffsetLiteral)
                    and isinstance(distance.value, int)
                ):
                    raise _UnsupportedScanInput()
                key = str(expr)
                if key not in added:
                    fresh_name = ir_misc.unique_symbol("__scan_input_shift", used_names)
                    used_names.add(fresh_name)
                    value = im.as_fieldop(im.lambda_(name)(im.deref(expr)), domain)(arguments[name])
                    added[key] = fresh_name, value
                return im.ref(added[key][0])

        body = ReplaceShifts().visit(stencil.expr)
        if not added:
            return node
        # Lifted reads can leave the original field unused. In particular, a
        # full-level field cannot remain a scalar input to a half-level scan.
        referenced_names = set(symbol_ref_utils.collect_symbol_refs(body))
        inputs = [
            (str(param.id), arg)
            for param, arg in zip(stencil.params, node.args, strict=True)
            if str(param.id) in referenced_names
        ] + list(added.values())
        new_scan = restore_scan(im.lambda_(*(name for name, _ in inputs))(body))
        return im.as_fieldop(new_scan, node.fun.args[1])(*(arg for _, arg in inputs))


def normalize_scan_producers(
    program: ir.Program, *, offset_provider_type: common.OffsetProviderType
) -> ir.Program:
    """Combine scan-input expressions within their original function scope.

    Run before function inlining and field-view normalization. Function
    parameters and calls to other field operators remain boundaries. Only
    callers of simple scan wrappers are considered. Local arithmetic bindings
    may be expanded, but bindings containing another scan or a called field
    operator are preserved. The later one-layer scan fusion can then absorb
    complete coefficients without absorbing their caller's upstream work.

    This is an experimental preparation policy, not a profitability guarantee.
    It changes producer expressions but leaves scan fusion to `fuse_scan_inputs`,
    where output-alias checks, input selection and shift lifting still apply.
    """
    scan_names = {
        str(function.id)
        for function in program.function_definitions
        if cpm.is_applied_as_fieldop(function.expr)
        and cpm.is_call_to(function.expr.fun.args[0], "scan")
        and len(function.expr.args) == len(function.params)
        and all(
            isinstance(arg, ir.SymRef) and arg.id == param.id
            for arg, param in zip(function.expr.args, function.params, strict=True)
        )
    }
    if not scan_names:
        return program
    result = inference.infer(program, offset_provider_type=offset_provider_type)

    def is_local_arithmetic(expr: ir.Expr) -> bool:
        if isinstance(expr, (ir.SymRef, ir.Literal)):
            return True
        if _is_non_scan_fieldop(expr):
            return all(is_local_arithmetic(arg) for arg in expr.args)
        return isinstance(expr.type, ts.ScalarType)

    class NormalizeProducer(eve.NodeTranslator):
        def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.Expr:
            if not _is_non_scan_fieldop(node):
                return node
            args = [self.visit(arg) for arg in node.args]
            node = inference.infer(
                ir_misc.canonicalize_as_fieldop(im.call(node.fun)(*args)),
                offset_provider_type=offset_provider_type,
                allow_undeclared_symbols=True,
            )
            assert cpm.is_applied_as_fieldop(node)
            # An empty set means unused; {()} means a read without a shift.
            return fuse_as_fieldop.FuseAsFieldOp.fuse_once(
                node,
                arg_predicate=lambda arg, accesses: (
                    _is_non_scan_fieldop(arg) and accesses in (set(), {()})
                ),
                offset_provider_type=offset_provider_type,
                enable_cse=False,
                uids=utils.IDGeneratorPool(),
            )

    class PrepareCall(eve.NodeTranslator):
        def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.Expr:
            bindings = kwargs.get("bindings", {})
            if cpm.is_let(node):
                args = [self.visit(arg, bindings=bindings) for arg in node.args]
                expanded = [
                    remap_symbols.RemapSymbolRefs().visit(arg, symbol_map=bindings) for arg in args
                ]
                parameters = {
                    str(param.id): arg for param, arg in zip(node.fun.params, expanded, strict=True)
                }
                local = {
                    name: value for name, value in bindings.items() if name not in parameters
                } | {name: arg for name, arg in parameters.items() if is_local_arithmetic(arg)}
                body = self.visit(node.fun.expr, bindings=local)
                return im.call(im.lambda_(*node.fun.params)(body))(*args)
            if isinstance(node.fun, ir.SymRef) and str(node.fun.id) in scan_names:
                return im.call(node.fun)(
                    *(
                        NormalizeProducer().visit(
                            remap_symbols.RemapSymbolRefs().visit(arg, symbol_map=bindings)
                        )
                        for arg in node.args
                    )
                )
            return self.generic_visit(node, **kwargs)

        def visit_Lambda(
            self, node: ir.Lambda, *, bindings: dict[str, ir.Expr], **kwargs: Any
        ) -> ir.Lambda:
            parameters = {str(param.id) for param in node.params}
            bindings = {name: value for name, value in bindings.items() if name not in parameters}
            return ir.Lambda(params=node.params, expr=self.visit(node.expr, bindings=bindings))

    for function in result.function_definitions:
        if any(
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.SymRef)
            and str(node.fun.id) in scan_names
            for node in function.expr.pre_walk_values()
        ):
            function.expr = PrepareCall().visit(function.expr, bindings={})
    return result


def fuse_scan_inputs(
    program: ir.Program,
    *,
    offset_provider_type: common.OffsetProviderType,
) -> ir.Program:
    """Fuse immediate producer expressions into scans.

    Fuse one immediate producer layer using the existing field-op helper.
    All eligible immediate producers are considered; newly exposed inputs
    are never revisited. Eligibility checks establish safety, not profitability.
    DaCe scan parameters are values at the current level, so fixed vertical
    shifts introduced by fusion must become separate shifted field arguments.
    The backend can then eliminate those copies instead of materializing full
    coefficient fields. Scan direction, initial state and arithmetic casts are
    retained. Unsupported shifted accesses keep the original scan.

    Input and output fields with different symbols must not alias at runtime.
    This pass checks overlap through IR symbols and let bindings; it cannot
    inspect the storage supplied by the caller.

    Args:
        program: Typed, domain-inferred iterator IR before SDFG lowering.
        offset_provider_type: Connectivity types used for type inference.

    Returns:
        A program with eligible scan inputs fused; unrelated assignments remain.
        After a rewrite, the caller must normalize the new field expressions
        and infer their domains before lowering.
    """

    def reads(expr: ir.Expr, aliases: dict[str, set[str]]) -> set[str]:
        return set().union(
            *(aliases.get(ref, {ref}) for ref in symbol_ref_utils.collect_symbol_refs(expr))
        )

    class FuseStatements(eve.NodeTranslator):
        def visit_SetAt(self, node: ir.SetAt, **kwargs: Any) -> ir.SetAt:
            targets = reads(node.target, {})
            return ir.SetAt(
                expr=self.visit(node.expr, targets=targets, aliases={}),
                domain=node.domain,
                target=node.target,
            )

        def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.Expr:
            targets = kwargs.get("targets", set())
            aliases = kwargs.get("aliases", {})
            if cpm.is_let(node):
                assert isinstance(node.fun, ir.Lambda)
                args = self.visit(node.args, **kwargs)
                bindings = {
                    str(param.id): reads(arg, aliases)
                    for param, arg in zip(node.fun.params, args, strict=True)
                }
                body = self.visit(node.fun.expr, **{**kwargs, "aliases": aliases | bindings})
                if body == node.fun.expr and args == node.args:
                    return node
                updated = im.call(im.lambda_(*(param.id for param in node.fun.params))(body))(*args)
                # Clean up lets without revisiting the newly exposed inputs.
                return inline_lambdas.inline_lambda(updated, opcount_preserving=True)
            if not (cpm.is_applied_as_fieldop(node) and cpm.is_call_to(node.fun.args[0], "scan")):
                return self.generic_visit(node, **kwargs)
            # Only reads moved into this recurrence matter. Unrelated outputs
            # of the enclosing assignment can be updated in place. Follow let
            # bindings so aliases cannot hide an input/output overlap.
            if any(reads(arg, aliases) & targets for arg in node.args):
                return node
            fused = fuse_as_fieldop.FuseAsFieldOp.fuse_once(
                node,
                arg_predicate=lambda arg, _: _is_non_scan_fieldop(arg),
                offset_provider_type=offset_provider_type,
                uids=utils.IDGeneratorPool(),
            )
            if fused is node:
                return node
            try:
                return _LiftScanShifts().visit(fused)
            except _UnsupportedScanInput:
                return node

    result = FuseStatements().visit(program)
    if result == program:
        return program
    return inference.infer(result, offset_provider_type=offset_provider_type)
