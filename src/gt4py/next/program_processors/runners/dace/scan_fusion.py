# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fuse scan inputs while preserving DaCe's scalar scan-argument convention."""

import copy
from collections.abc import Callable
from typing import Any

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import builtins, ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import (
    fuse_as_fieldop,
    inline_lambdas,
    pass_manager,
    trace_shifts,
)
from gt4py.next.iterator.type_system import inference
from gt4py.next.type_system import type_specifications as ts


class _UnsupportedScanInput(Exception):
    """Keep the original statement when a fused scan needs unsupported accesses."""


class _LiftScanShifts(eve.NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.FunCall:
        if not (cpm.is_applied_as_fieldop(node) and cpm.is_call_to(node.fun.args[0], "scan")):
            return self.generic_visit(node, **kwargs)
        scan = node.fun.args[0]
        stencil, forward, init = scan.args
        if not isinstance(stencil, ir.Lambda) or len(node.fun.args) != 2:
            raise _UnsupportedScanInput()
        domain = node.fun.args[1]
        arguments = dict(zip((str(p.id) for p in stencil.params[1:]), node.args, strict=True))
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
        uids = utils.IDGeneratorPool()

        class ReplaceShifts(eve.NodeTranslator):
            def visit_FunCall(self, expr: ir.FunCall, **kwargs: Any) -> ir.Expr:
                if not cpm.is_applied_shift(expr):
                    return self.generic_visit(expr, **kwargs)
                # Initially support only fixed vertical shifts of direct scan inputs.
                if not isinstance(expr.args[0], ir.SymRef):
                    raise _UnsupportedScanInput()
                name = str(expr.args[0].id)
                offsets = expr.fun.args
                if name not in arguments or len(offsets) != 2:
                    raise _UnsupportedScanInput()
                axis, distance = offsets
                if not (
                    isinstance(axis, ir.CartesianOffset)
                    and axis.domain == axis.codomain
                    and axis.domain.kind == common.DimensionKind.VERTICAL
                    and isinstance(distance, ir.OffsetLiteral)
                    and isinstance(distance.value, int)
                ):
                    raise _UnsupportedScanInput()
                key = str(expr)
                if key not in added:
                    fresh_name = next(uids["__scan_input_shift"])
                    while fresh_name in used_names:
                        fresh_name = next(uids["__scan_input_shift"])
                    used_names.add(fresh_name)
                    value = im.as_fieldop(im.lambda_(name)(im.deref(expr)), domain)(arguments[name])
                    added[key] = fresh_name, value
                return im.ref(added[key][0])

        body = ReplaceShifts().visit(stencil.expr)
        if not added:
            return node
        params = [p.id for p in stencil.params] + [p for p, _ in added.values()]
        new_scan = im.call("scan")(im.lambda_(*params)(body), forward, init)
        return im.as_fieldop(new_scan, node.fun.args[1])(
            *node.args, *(value for _, value in added.values())
        )


def normalize_scan_producers(
    program: ir.Program, *, offset_provider: common.OffsetProvider
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
    offset_provider_type = common.offset_provider_to_type(offset_provider)
    result = inference.infer(copy.deepcopy(program), offset_provider_type=offset_provider_type)

    def is_local_arithmetic(expr: ir.Expr) -> bool:
        if isinstance(expr, (ir.SymRef, ir.Literal)):
            return True
        if cpm.is_applied_as_fieldop(expr) and not cpm.is_call_to(expr.fun.args[0], "scan"):
            return all(is_local_arithmetic(arg) for arg in expr.args)
        return isinstance(expr.type, ts.ScalarType)

    class ExpandBindings(eve.NodeTranslator):
        def visit_SymRef(self, node: ir.SymRef, **kwargs: Any) -> ir.Expr:
            return kwargs["bindings"].get(str(node.id), node)

        def visit_Lambda(self, node: ir.Lambda, **kwargs: Any) -> ir.Lambda:
            bindings = {
                name: value
                for name, value in kwargs["bindings"].items()
                if name not in {str(param.id) for param in node.params}
            }
            return ir.Lambda(params=node.params, expr=self.visit(node.expr, bindings=bindings))

    class NormalizeProducer(eve.NodeTranslator):
        def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.Expr:
            if not (
                cpm.is_applied_as_fieldop(node) and not cpm.is_call_to(node.fun.args[0], "scan")
            ):
                return node
            args = [self.visit(arg) for arg in node.args]
            node = inference.infer(
                ir_misc.canonicalize_as_fieldop(im.call(node.fun)(*args)),
                offset_provider_type=offset_provider_type,
                allow_undeclared_symbols=True,
            )
            assert cpm.is_applied_as_fieldop(node)
            shifts = trace_shifts.trace_stencil(node.fun.args[0], num_args=len(args))
            eligible = [
                cpm.is_applied_as_fieldop(arg)
                and not cpm.is_call_to(arg.fun.args[0], "scan")
                and accesses in (set(), {()})
                for arg, accesses in zip(args, shifts, strict=True)
            ]
            if not any(eligible):
                return node
            return fuse_as_fieldop.fuse_as_fieldop(
                node,
                eligible_args=eligible,
                offset_provider_type=offset_provider_type,
                enable_cse=False,
                uids=utils.IDGeneratorPool(),
            )

    class PrepareCall(eve.NodeTranslator):
        def visit_FunCall(self, node: ir.FunCall, **kwargs: Any) -> ir.Expr:
            bindings = kwargs.get("bindings", {})
            if cpm.is_let(node):
                args = [self.visit(arg, bindings=bindings) for arg in node.args]
                expanded = [ExpandBindings().visit(arg, bindings=bindings) for arg in args]
                local = {
                    name: value
                    for name, value in bindings.items()
                    if name not in {str(param.id) for param in node.fun.params}
                } | {
                    str(param.id): arg
                    for param, arg in zip(node.fun.params, expanded, strict=True)
                    if is_local_arithmetic(arg)
                }
                body = self.visit(node.fun.expr, bindings=local)
                return im.call(im.lambda_(*node.fun.params)(body))(*args)
            if isinstance(node.fun, ir.SymRef) and str(node.fun.id) in scan_names:
                return im.call(node.fun)(
                    *(
                        NormalizeProducer().visit(
                            inference.infer(
                                ExpandBindings().visit(arg, bindings=bindings),
                                offset_provider_type=offset_provider_type,
                                allow_undeclared_symbols=True,
                            )
                        )
                        for arg in node.args
                    )
                )
            return self.generic_visit(node, **kwargs)

        def visit_Lambda(self, node: ir.Lambda, **kwargs: Any) -> ir.Lambda:
            bindings = {
                name: value
                for name, value in kwargs.get("bindings", {}).items()
                if name not in {str(param.id) for param in node.params}
            }
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
    offset_provider: common.OffsetProvider,
    input_selector: Callable[[ir.FunCall, int], bool] | None = None,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> ir.Program:
    """Inline coefficient expressions into scans without changing model code.

    Fuse one immediate producer layer using the existing field-op helper.
    The optional selector chooses input indices on the original scan call;
    the carry is not an input index. Newly exposed inputs are never revisited.
    By default all eligible immediate producers are selected; this is not a
    profitability heuristic. Unselected producers retain their computation.
    DaCe scan parameters are values at the current level, so fixed vertical
    shifts introduced by fusion must become separate shifted field arguments.
    The backend can then eliminate those copies instead of materializing full
    coefficient fields. Scan direction, initial state and arithmetic casts are
    retained. Unsupported shifted accesses keep the original scan.

    Args:
        program: Typed, domain-inferred iterator IR before SDFG lowering.
        offset_provider: Connectivities used for type and domain inference.
        input_selector: Optional predicate of the original scan call and input
            index. Selection cannot override fusion safety checks.
        use_max_domain_range_on_unstructured_shift: Preserve the backend domain policy.

    Returns:
        A program with eligible scan inputs fused; unrelated assignments remain.
    """

    offset_provider_type = common.offset_provider_to_type(offset_provider)

    def reads(expr: ir.Expr, aliases: dict[str, set[str]]) -> set[str]:
        return set().union(
            *(
                aliases.get(str(ref.id), {str(ref.id)})
                for ref in expr.pre_walk_values()
                if isinstance(ref, ir.SymRef) and ref.id not in builtins.BUILTINS
            )
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
            shifts = trace_shifts.trace_stencil(node.fun.args[0], num_args=len(node.args))
            eligible = [
                cpm.is_applied_as_fieldop(arg)
                and not cpm.is_call_to(arg.fun.args[0], "scan")
                and fuse_as_fieldop._arg_inline_predicate(arg, accesses)
                and (input_selector is None or input_selector(node, index))
                for index, (arg, accesses) in enumerate(zip(node.args, shifts, strict=True))
            ]
            if not any(eligible):
                return node
            fused = fuse_as_fieldop.fuse_as_fieldop(
                copy.deepcopy(node),
                eligible_args=eligible,
                offset_provider_type=offset_provider_type,
                uids=utils.IDGeneratorPool(),
                enable_cse=True,
            )
            try:
                return _LiftScanShifts().visit(fused)
            except _UnsupportedScanInput:
                return node

    result = FuseStatements().visit(program)
    if result == program:
        return program
    result = inference.infer(result, offset_provider_type=offset_provider_type)
    return pass_manager.apply_fieldview_transforms(
        result,
        offset_provider=offset_provider,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )
