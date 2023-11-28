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
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref
from gt4py.next.iterator.transforms.scan_eta_reduction import ScanEtaReduction
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
            | InlineLifts.Flag.INLINE_LIFTED_ARGS
            # needed for UnrollReduce and lift args like `(↑(λ() → constant)`
        ).visit(ir)
    else:
        raise ValueError()

    return ir


def _inline_into_scan(ir, *, max_iter=10):
    for _ in range(10):
        # in case there are multiple levels of lambdas around the scan we have to do multiple iterations
        inlined = InlineIntoScan().visit(ir)
        inlined = InlineLambdas.apply(inlined, opcount_preserving=True, force_inline_lift_args=True)
        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError(f"Inlining into scan did not converge with {max_iter} iterations.")
    return ir


def apply_common_transforms(
    ir: ir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    unconditionally_collapse_tuples=False,
):
    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    ir = MergeLet().visit(ir)
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = PropagateDeref.apply(ir)
    ir = NormalizeShifts().visit(ir)

    for _ in range(10):
        inlined = ir

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
        inlined = CollapseTuple.apply(
            inlined,
            # to limit number of times global type inference is executed, only in the last iterations.
            use_global_type_inference=inlined == ir,
        )

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining lift and lambdas did not converge.")

    # Since `CollapseTuple` relies on the type inference which does not support returning tuples
    # larger than the number of closure outputs as given by the unconditional collapse, we can
    # only run the unconditional version here instead of in the loop above.
    if unconditionally_collapse_tuples:
        ir = CollapseTuple.apply(ir, ignore_tuple_size=unconditionally_collapse_tuples)

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
            ir = _inline_lifts(ir, lift_mode)
            ir = NormalizeShifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed.")

    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)
        ir = InlineLifts().visit(ir)
        # If after creating temporaries, the scan is not at the top, we inline.
        # The following example doesn't have a lift around the shift, i.e. temporary pass will not extract it.
        # λ(inp) → scan(λ(state, k, kp) → state + ·k + ·kp, True, 0.0)(inp, ⟪Koffₒ, 1ₒ⟫(inp))`
        ir = _inline_into_scan(ir)

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
