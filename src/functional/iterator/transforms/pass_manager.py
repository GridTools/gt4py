import enum

from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
from functional.iterator.transforms.inline_tuple_get import InlineTupleGet
from functional.iterator.transforms.normalize_shifts import NormalizeShifts
from functional.iterator.transforms.shift_transformer import (
    PropagateShiftTransformer,
    RemoveShiftsTransformer,
)
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
):
    if ir.id == "__field_operator_fvm_advect":
        breakpoint()

    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    assert isinstance(lift_mode, LiftMode)
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode != LiftMode.FORCE_TEMPORARIES:
        for _ in range(10):
            inlined = _inline_lifts(ir, lift_mode)
            inlined = InlineLambdas.apply(inlined)
            inlined = InlineTupleGet.apply(inlined)
            inlined = RemoveShiftsTransformer.apply(inlined)
            inlined = PropagateShiftTransformer.apply(inlined)
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift and lambdas did not converge.")
    else:
        ir = InlineLambdas.apply(ir)

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


    #if ir.id == "__field_operator_fvm_advect":
    #    ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)

    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)
        ir = InlineLifts().visit(ir)

    ir = EtaReduction().visit(ir)

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination().visit(ir)

    ir = InlineLambdas.apply(ir, opcount_preserving=common_subexpression_elimination)

    return ir
