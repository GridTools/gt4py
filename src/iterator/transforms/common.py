from iterator.transforms.global_tmps import CreateGlobalTmps
from iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from iterator.transforms.inline_lambdas import InlineLambdas
from iterator.transforms.inline_lifts import InlineLifts
from iterator.transforms.normalize_shifts import NormalizeShifts


def apply_common_transforms(ir, use_tmps=False, offset_provider=None, register_tmp=None):
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if not use_tmps:
        ir = InlineLifts().visit(ir)
    ir = InlineLambdas().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if use_tmps:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(
            ir, offset_provider=offset_provider, register_tmp=register_tmp
        )
    return ir
