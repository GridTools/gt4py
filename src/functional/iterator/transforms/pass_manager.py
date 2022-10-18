import enum

from functional.iterator import ir
from functional.iterator.transforms.cse import CommonSubexpressionElimination
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.global_tmps import CreateGlobalTmps
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.iterator.transforms.inline_lifts import InlineLifts
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


from eve import NOTHING, NodeTranslator
from functional.iterator import ir
from eve.pattern_matching import ObjectPattern as P
class RemoveShiftsTransformer(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        # deref(ignore_shift(...)(it)) -> deref(it)
        if P(ir.FunCall,
            fun=ir.SymRef(id="deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))]
          ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"), args=[node.args[0].args[0]]))

        # deref(translate_shift(...)(it)) -> deref(it)
        if P(ir.FunCall,
             fun=ir.SymRef(id="deref"),
             args=[P(ir.FunCall,
                     fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))]
             ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"),
                                         args=[node.args[0].args[0]]))

        # can_deref(ignore_shift(...)(it)) -> deref(it)
        if P(ir.FunCall,
            fun=ir.SymRef(id="can_deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))]
          ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[node.args[0].args[0]]))

        # can_deref(translate_shift(...)(it)) -> deref(it)
        if P(ir.FunCall,
             fun=ir.SymRef(id="can_deref"),
             args=[P(ir.FunCall,
                     fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))]
             ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="can_deref"),
                                         args=[node.args[0].args[0]]))

        return self.generic_visit(node)

class PropagateShiftTransformer(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        # shift(...)(translate_shift(...)(it)) -> translate_shift(...)(shift(...)(it))
        if P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="shift")), args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))]).match(node):
            assert len(node.fun.args) == 2
            shift_tag, shift_index = node.fun.args
            old_tag, new_tag = node.args[0].fun.args
            if old_tag == shift_tag:
                shift_tag = new_tag

            new_shift = ir.FunCall(fun=ir.SymRef(id="shift"), args=[shift_tag, shift_index])
            translate_shift = node.args[0].fun
            it = node.args[0].args

            return ir.FunCall(fun=translate_shift, args=[self.visit(ir.FunCall(fun=new_shift, args=it))])
        # shift(...)(ignore_shift(...)(it)) -> ignore_shift(...)(shift(...)(it))
        elif P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="shift")), args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))]).match(node):
            assert len(node.fun.args) == 2
            shift_tag, shift_index = node.fun.args
            ignored_tag = node.args[0].fun.args[0]
            if ignored_tag == shift_tag:
                return node.args[0]

            shift = node.fun
            it = node.args[0].args
            ignore_shift = node.args[0].fun

            return ir.FunCall(fun=ignore_shift, args=[self.visit(ir.FunCall(fun=shift, args=it))])
        return node

class InlineTupleAccess(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        if P(ir.FunCall, fun=ir.SymRef(id="tuple_get")).match(node):
            index, tuple_ = node.args
            if P(ir.FunCall, fun=ir.SymRef(id="make_tuple")).match(tuple_):
                assert isinstance(index, ir.Literal) and index.type == "int"
                return self.generic_visit(tuple_.args[int(index.value)])
        return self.generic_visit(node)

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
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    if lift_mode != LiftMode.FORCE_TEMPORARIES:
        for _ in range(30):
            inlined = _inline_lifts(ir, lift_mode)
            inlined = InlineLambdas.apply(inlined)
            inlined = InlineTupleAccess().visit(inlined)
            inlined = RemoveShiftsTransformer().visit(inlined)
            inlined = PropagateShiftTransformer().visit(inlined)
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
            inlined = RemoveShiftsTransformer().visit(inlined)
            inlined = PropagateShiftTransformer().visit(inlined)
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

    ir = InlineLambdas.apply(ir, opcount_preserving=common_subexpression_elimination)

    return ir
