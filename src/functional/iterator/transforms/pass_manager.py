import enum

from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
from functional.iterator.transforms.merge_let import MergeLet
from functional.iterator.transforms.normalize_shifts import NormalizeShifts
from functional.iterator.transforms.simple_inline_heuristic import heuristic
from functional.iterator.transforms.unroll_reduce import UnrollReduce


@enum.unique
class LiftMode(enum.Enum):
    FORCE_INLINE = enum.auto()
    FORCE_TEMPORARIES = enum.auto()
    SIMPLE_HEURISTIC = enum.auto()


def _inline_lifts(ir, lift_mode):
    if lift_mode == LiftMode.FORCE_INLINE:
        return InlineLifts().visit(ir)
    if lift_mode == LiftMode.SIMPLE_HEURISTIC:
        predicate = heuristic(ir)
        return InlineLifts(predicate).visit(ir)
    assert lift_mode == LiftMode.FORCE_TEMPORARIES
    return ir


def apply_common_transforms(
    ir: ir.Node,
    *,
    lift_mode=None,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lift=False,
):
    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    ir = MergeLet().visit(ir)
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode != LiftMode.FORCE_TEMPORARIES:
        for _ in range(10):
            inlined = _inline_lifts(ir, lift_mode)
            inlined = InlineLambdas.apply(
                inlined, opcount_preserving=True, force_inline_lift=force_inline_lift
            )
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift and lambdas did not converge.")
    else:
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    ir = NormalizeShifts().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider=offset_provider)
            if unrolled == ir:
                break
            ir = unrolled
            ir = NormalizeShifts().visit(ir)
            ir = _inline_lifts(ir, lift_mode)
            ir = NormalizeShifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed.")
    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)
        ir = InlineLifts().visit(ir)

    ir = EtaReduction().visit(ir)

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination().visit(ir)
        ir = MergeLet().visit(ir)

    ir = InlineLambdas.apply(ir, opcount_preserving=True, force_inline_lift=force_inline_lift)

    return ir
