import enum

from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
from functional.iterator.transforms.normalize_shifts import NormalizeShifts
from functional.iterator.transforms.simple_inline_heuristic import heuristic


@enum.unique
class LiftMode(enum.Enum):
    FORCE_INLINE = enum.auto()
    FORCE_TEMPORARIES = enum.auto()
    SIMPLE_HEURISTIC = enum.auto()


def apply_common_transforms(ir, lift_mode=None, offset_provider=None):
    if lift_mode is None:
        lift_mode = LiftMode.FORCE_INLINE
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode == LiftMode.FORCE_INLINE:
        ir = InlineLifts().visit(ir)
    elif lift_mode == LiftMode.SIMPLE_HEURISTIC:
        predicate = heuristic(ir)
        ir = InlineLifts(predicate).visit(ir)
    ir = InlineLambdas().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode != LiftMode.FORCE_INLINE:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(ir, offset_provider=offset_provider)
    return ir
