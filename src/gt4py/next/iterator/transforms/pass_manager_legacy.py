# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# FIXME[#1582](tehrengruber): file should be removed after refactoring to GTIR
import enum
from typing import Callable, Optional

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import fencil_to_program, inline_fundefs
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.eta_reduction import EtaReduction
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_center_deref_lift_vars import InlineCenterDerefLiftVars
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
    USE_TEMPORARIES = enum.auto()


def _inline_lifts(ir, lift_mode):
    if lift_mode == LiftMode.FORCE_INLINE:
        return InlineLifts().visit(ir)
    elif lift_mode == LiftMode.USE_TEMPORARIES:
        return InlineLifts(
            flags=InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT
            | InlineLifts.Flag.INLINE_DEREF_LIFT  # some tuple exprs found in FVM don't work yet.
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
        raise RuntimeError(f"Inlining into 'scan' did not converge within {max_iter} iterations.")
    return ir


def apply_common_transforms(
    ir: itir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    unconditionally_collapse_tuples=False,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    offset_provider_type: Optional[common.OffsetProviderType] = None,
) -> itir.Program:
    assert isinstance(ir, itir.FencilDefinition)
    # TODO(havogt): if the runtime `offset_provider` is not passed, we cannot run global_tmps
    if offset_provider_type is None:
        offset_provider_type = common.offset_provider_to_type(offset_provider)

    ir = fencil_to_program.FencilToProgram().apply(ir)
    icdlv_uids = eve_utils.UIDGenerator()

    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)  # type: ignore[arg-type] # all previous passes return itir.Program
    ir = PropagateDeref.apply(ir)
    ir = NormalizeShifts().visit(ir)

    for _ in range(10):
        inlined = ir

        inlined = InlineCenterDerefLiftVars.apply(inlined, uids=icdlv_uids)  # type: ignore[type-var]  # always a fencil
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
            offset_provider_type=offset_provider_type,
            # TODO(tehrengruber): disabled since it increases compile-time too much right now
            flags=~CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES,
        )
        # This pass is required such that a deref outside of a
        # `tuple_get(make_tuple(let(...), ...))` call is propagated into the let after the
        # `tuple_get` is removed by the `CollapseTuple` pass.
        inlined = PropagateDeref.apply(inlined)

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    if lift_mode != LiftMode.FORCE_INLINE:
        raise NotImplementedError()

    # Since `CollapseTuple` relies on the type inference which does not support returning tuples
    # larger than the number of closure outputs as given by the unconditional collapse, we can
    # only run the unconditional version here instead of in the loop above.
    if unconditionally_collapse_tuples:
        ir = CollapseTuple.apply(
            ir,
            ignore_tuple_size=True,
            offset_provider_type=offset_provider_type,
            # TODO(tehrengruber): disabled since it increases compile-time too much right now
            flags=~CollapseTuple.Flag.PROPAGATE_TO_IF_ON_TUPLES,
        )

    if lift_mode == LiftMode.FORCE_INLINE:
        ir = _inline_into_scan(ir)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps().visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider_type=offset_provider_type)
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
        ir = CommonSubexpressionElimination.apply(ir, offset_provider_type=offset_provider_type)  # type: ignore[type-var]  # always an itir.Program
        ir = MergeLet().visit(ir)

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir
