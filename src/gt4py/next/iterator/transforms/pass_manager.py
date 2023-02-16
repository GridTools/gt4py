# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction
from gt4py.next.iterator.transforms.global_tmps import CreateGlobalTmps
from gt4py.next.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.inline_tuple_get import InlineTupleGet
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref
from gt4py.next.iterator.transforms.shift_transformer import (
    PropagateShiftTransformer,
    RemoveShiftsTransformer,
)
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce


@enum.unique
class LiftMode(enum.Enum):
    FORCE_INLINE = enum.auto()
    FORCE_TEMPORARIES = enum.auto()
    SIMPLE_HEURISTIC = enum.auto()


def _inline_lifts(ir, lift_mode):
    if lift_mode == LiftMode.FORCE_INLINE:
        return InlineLifts().visit(ir)
    if lift_mode == LiftMode.SIMPLE_HEURISTIC:
        return InlineLifts(simple_inline_heuristic.is_eligible_for_inlining).visit(ir)
    assert lift_mode == LiftMode.FORCE_TEMPORARIES
    return ir


def apply_common_transforms(
    ir: ir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
):
    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    ir = MergeLet().visit(ir)
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = PropagateDeref.apply(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode != LiftMode.FORCE_TEMPORARIES:
        for _ in range(10):
            inlined = _inline_lifts(ir, lift_mode)
            inlined = InlineLambdas.apply(
                inlined,
                opcount_preserving=True,
                force_inline_lift=(lift_mode == LiftMode.FORCE_INLINE),
            )
            inlined = PropagateDeref.apply(inlined)
            inlined = InlineTupleGet.apply(inlined)
            inlined = RemoveShiftsTransformer.apply(inlined)
            inlined = PropagateShiftTransformer.apply(inlined)
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift and lambdas did not converge.")
    else:
        ir = InlineLambdas.apply(
            ir, opcount_preserving=True, force_inline_lift=(lift_mode == LiftMode.FORCE_INLINE)
        )

    ir = NormalizeShifts().visit(ir)

    if unroll_reduce:
        ir = UnrollReduce.apply(ir, offset_provider=offset_provider)
        for _ in range(10):
            inlined = NormalizeShifts().visit(ir)
            inlined = _inline_lifts(inlined, lift_mode)
            inlined = InlineLambdas.apply(inlined, opcount_preserving=True, force_inline_lift=True)
            inlined = NormalizeShifts().visit(inlined)
            inlined = RemoveShiftsTransformer.apply(inlined)
            inlined = PropagateShiftTransformer.apply(inlined)
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift did not converge.")

    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)
        ir = InlineLifts().visit(ir)

    ir = EtaReduction().visit(ir)

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination().visit(ir)
        ir = MergeLet().visit(ir)

    ir = InlineLambdas.apply(ir, opcount_preserving=True)

    return ir
