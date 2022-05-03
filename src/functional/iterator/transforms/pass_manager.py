from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
from functional.iterator.transforms.normalize_shifts import NormalizeShifts
from functional.iterator.transforms.unroll_reduce import UnrollReduce


def apply_common_transforms(
    ir,
    use_tmps=False,
    offset_provider=None,
    register_tmp=None,
    unroll_reduce=False,
):
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    ir = InlineLambdas().visit(ir)
    if not use_tmps:
        ir = InlineLifts().visit(ir)
    ir = InlineLambdas().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce().visit(ir, offset_provider=offset_provider)
            if unrolled == ir:
                break
            ir = unrolled
            ir = NormalizeShifts().visit(ir)
            if not use_tmps:
                ir = InlineLifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed")
    if use_tmps:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(
            ir, offset_provider=offset_provider, register_tmp=register_tmp
        )
    return ir
