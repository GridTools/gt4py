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

import enum
from typing import Callable, Optional

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms import simple_inline_heuristic
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.global_tmps import CreateGlobalTmps
from gt4py.next.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from gt4py.next.iterator.transforms.inline_into_scan import InlineIntoScan
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas, inline_lambda
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref
from gt4py.next.iterator.transforms.scan_eta_reduction import ScanEtaReduction
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce


@enum.unique
class LiftMode(enum.Enum):
    FORCE_INLINE = enum.auto()
    FORCE_TEMPORARIES = enum.auto()
    SIMPLE_HEURISTIC = enum.auto()


def _inline_lifts(ir, lift_mode):
    if lift_mode == LiftMode.FORCE_INLINE:
        return InlineLifts().visit(ir)
    elif lift_mode == LiftMode.SIMPLE_HEURISTIC:
        return InlineLifts(simple_inline_heuristic.is_eligible_for_inlining).visit(ir)
    elif lift_mode == LiftMode.FORCE_TEMPORARIES:
        return InlineLifts(
            flags=InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT
            | InlineLifts.Flag.INLINE_DEREF_LIFT  # some tuple exprs found in FVM don't work yet.
            | InlineLifts.Flag.INLINE_CENTRE_ONLY_LIFT_ARGS
        ).visit(ir)
    else:
        raise ValueError()

    return ir


def _inline_into_scan(ir, *, max_iter=10):
    for _ in range(10):
        # in case there are multiple levels of lambdas around the scan we have to do multiple iterations
        inlined = InlineIntoScan().visit(ir)
        inlined = InlineLambdas.apply(inlined, opcount_preserving=True, force_inline_lift_args=False)
        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError(f"Inlining into 'scan' did not converge within {max_iter} iterations.")
    return ir

import gt4py.next.iterator.ir_utils.common_pattern_matcher as common_pattern_matcher
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py import eve

class EnsureNoLiftCapture(eve.VisitorWithSymbolTableTrait):
    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        self.generic_visit(node, **kwargs)
        if common_pattern_matcher.is_applied_lift(node):
            stencil = node.fun.args[0]
            used_symbols = collect_symbol_refs(stencil)
            if used_symbols:
                raise "123"


unique_id = 0

from gt4py.next.iterator.ir_utils import ir_makers as im

class InlineSinglePosDerefLiftArgs(eve.NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("recorded_shifts",)

    def visit_StencilClosure(self, node: ir.StencilClosure):
        TraceShifts.apply(node, save_to_annex=True)
        return self.generic_visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            eligible_params = [False] * len(node.fun.params)

            # force inline lift args derefed at at most a single position
            new_args = []
            bound_scalars = {}
            # TODO: what is node.fun is not a lambda? e.g. directly deref?
            for i, (param, arg) in enumerate(zip(node.fun.params, node.args)):
                if common_pattern_matcher.is_applied_lift(arg) and not hasattr(param.annex, "recorded_shifts"):
                    breakpoint()
                if common_pattern_matcher.is_applied_lift(arg) and param.annex.recorded_shifts in [set(), {()}]:
                    eligible_params[i] = True
                    global unique_id
                    bound_arg_name = f"__wtf{unique_id}"
                    unique_id+=1
                    new_args.append(im.lift(im.lambda_()(bound_arg_name))())
                    bound_scalars[bound_arg_name] = InlineLifts(flags=InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT).visit(im.deref(arg), recurse=False)
                else:
                    new_args.append(arg)

            if any(eligible_params):
                # TODO(tehrengruber): propagate let outwards
                new_node = inline_lambda(
                    ir.FunCall(
                        fun=node.fun,
                        args=new_args
                    ),
                    eligible_params=eligible_params,
                )
                return im.let(*bound_scalars.items())(new_node)

        return node


def main_transforms(
    ir: ir.Node,
    lift_mode=None
):
    stage = 0
    for _ in range(10):
        inlined = ir

        # TODO: save trace shifts info here once and don't recompute twice below
        inlined = InlineSinglePosDerefLiftArgs().visit(inlined)
        inlined = _inline_lifts(inlined, lift_mode)

        inlined = InlineLambdas.apply(
            inlined,
            opcount_preserving=True,
            force_inline_lift_args=(lift_mode == LiftMode.FORCE_INLINE),
            # If trivial lifts are not inlined we might create temporaries for constants. In all
            #  other cases we want it anyway.
            force_inline_trivial_lift_args=True,
        )
        inlined = ConstantFolding.apply(inlined)
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        if stage == 1:
            inlined = CollapseTuple.apply(
                inlined,
                # to limit number of times global type inference is executed, only in the last iterations.
                # use_global_type_inference=inlined == ir,
                ignore_tuple_size=True,  # possibly dangerous
                use_global_type_inference=False,
            )
        inlined = PropagateDeref.apply(inlined)  # todo: document

        if inlined == ir:
            stage += 1
            if stage == 2:
                break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")
    return ir

# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `lift_mode` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: ir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    unconditionally_collapse_tuples=False,
    temporary_extraction_heuristics: Optional[
        Callable[[ir.StencilClosure], Callable[[ir.Expr], bool]]
    ] = None,
):
    lift_mode = LiftMode.FORCE_TEMPORARIES

    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    #ir = main_transforms(ir, lift_mode=lift_mode)
    ir = MergeLet().visit(ir)
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = PropagateDeref.apply(ir)
    ir = NormalizeShifts().visit(ir)

    #EnsureNoLiftCapture().visit(ir)
    #InlineLifts(flags=InlineLifts.Flag.INLINE_CENTRE_ONLY_LIFT_ARGS | InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT).visit(ir)
    #traced_shifts = TraceShifts.apply(ir.closures[0], inputs_only=False)

    ir = main_transforms(ir, lift_mode=lift_mode)

    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(
            ir,
            offset_provider=offset_provider,
            extraction_heuristics=temporary_extraction_heuristics,
        )
        ir = ConstantFolding.apply(ir)

        for _ in range(10):
            inlined = InlineLifts().visit(ir)
            inlined = InlineLambdas.apply(
                inlined,
                opcount_preserving=True,
                force_inline_lift_args=True,
            )
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift and lambdas did not converge.")

        # If after creating temporaries, the scan is not at the top, we inline.
        # The following example doesn't have a lift around the shift, i.e. temporary pass will not extract it.
        # λ(inp) → scan(λ(state, k, kp) → state + ·k + ·kp, True, 0.0)(inp, ⟪Koffₒ, 1ₒ⟫(inp))`
        ir = _inline_into_scan(ir)

    # Since `CollapseTuple` relies on the type inference which does not support returning tuples
    # larger than the number of closure outputs as given by the unconditional collapse, we can
    # only run the unconditional version here instead of in the loop above.
    #if unconditionally_collapse_tuples:
    #    ir = CollapseTuple.apply(ir, ignore_tuple_size=unconditionally_collapse_tuples)

    if lift_mode == LiftMode.FORCE_INLINE:
        ir = _inline_into_scan(ir)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps().visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider=offset_provider)
            if unrolled == ir:
                break
            ir = unrolled
            ir = CollapseListGet().visit(ir)
            ir = NormalizeShifts().visit(ir)
            ir = _inline_lifts(ir, LiftMode.FORCE_INLINE)
            ir = NormalizeShifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed.")

    ir = EtaReduction().visit(ir)
    ir = ScanEtaReduction().visit(ir)

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination().visit(ir)
        ir = MergeLet().visit(ir)

    ir = InlineLambdas.apply(
        ir,
        opcount_preserving=True,
        force_inline_lambda_args=force_inline_lambda_args,
    )

    return ir
