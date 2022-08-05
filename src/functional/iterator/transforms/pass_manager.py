from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination
from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
from functional.iterator.transforms.normalize_shifts import NormalizeShifts
from functional.iterator.transforms.unroll_reduce import UnrollReduce


def apply_common_transforms(
    ir: ir.Node,
    use_tmps=False,
    offset_provider=None,
    register_tmp=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
):
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if not use_tmps:
        for _ in range(10):
            inlined = InlineLifts().visit(ir)
            inlined = InlineLambdas.apply(inlined)
            if inlined == ir:
                break
            ir = inlined
        else:
            raise RuntimeError("Inlining lift and lambdas did not converge.")
    else:
        ir = InlineLambdas.apply(ir)

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
            raise RuntimeError("Reduction unrolling failed.")
    if use_tmps:
        assert offset_provider is not None
        ir = CreateGlobalTmps().visit(
            ir, offset_provider=offset_provider, register_tmp=register_tmp
        )

    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination().visit(ir)

    ir = InlineLambdas.apply(ir, opcount_preserving=common_subexpression_elimination)

    return ir
