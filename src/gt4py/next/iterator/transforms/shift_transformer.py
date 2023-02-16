from gt4py.eve import NodeTranslator
from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.iterator import ir


class RemoveShiftsTransformer(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        # deref(ignore_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))],
        ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"), args=[node.args[0].args[0]]))  # type: ignore[attr-defined]

        # deref(translate_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))],
        ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"), args=[node.args[0].args[0]]))  # type: ignore[attr-defined]

        # can_deref(ignore_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="can_deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))],
        ).match(node):
            return self.visit(
                ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[node.args[0].args[0]])  # type: ignore[attr-defined]
            )

        # can_deref(translate_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="can_deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))],
        ).match(node):
            return self.visit(
                ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[node.args[0].args[0]])  # type: ignore[attr-defined]
            )

        return self.generic_visit(node)


# Note that `ignore_shift` and `translate_shift` on a lifted stencil call do not propagate
# to the arguments as otherwise this:
#
# `deref(ignore_shift(V2EDimₒ)(↑(λ(it) → shift(V2EDimₒ, 1)(it)))(it))`
#
#  would be transformed into
#
# ⇔ `deref(↑(λ(it) → shift(V2EDimₒ, 1)(it))(ignore_shift(V2EDimₒ)(it)))`
# ⇔ `λ(it) → shift(V2EDimₒ, 1)(ignore_shift(V2EDimₒ)(it))`
# ⇔ `λ(it) → it`
class PropagateShiftTransformer(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        # shift(...)(translate_shift(...)(it)) -> translate_shift(...)(shift(...)(it))
        if P(
            ir.FunCall,
            fun=P(ir.FunCall, fun=ir.SymRef(id="shift")),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))],
        ).match(node):
            assert isinstance(node.fun, ir.FunCall) and len(node.fun.args) == 2
            assert len(node.args[0].fun.args) == 2  # type: ignore[attr-defined]
            shift_tag, shift_index = node.fun.args
            old_tag, new_tag = node.args[0].fun.args  # type: ignore[attr-defined]
            if old_tag == shift_tag:
                shift_tag = new_tag

            new_shift = ir.FunCall(fun=ir.SymRef(id="shift"), args=[shift_tag, shift_index])
            translate_shift = node.args[0].fun  # type: ignore[attr-defined]
            it = node.args[0].args  # type: ignore[attr-defined]

            return ir.FunCall(
                fun=translate_shift, args=[self.visit(ir.FunCall(fun=new_shift, args=it))]
            )
        # shift(...)(ignore_shift(...)(it)) -> ignore_shift(...)(shift(...)(it))
        elif P(
            ir.FunCall,
            fun=P(ir.FunCall, fun=ir.SymRef(id="shift")),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))],
        ).match(node):
            assert isinstance(node.fun, ir.FunCall) and len(node.fun.args) == 2
            assert len(node.args[0].fun.args) == 1  # type: ignore[attr-defined]
            shift_tag, shift_index = node.fun.args
            ignored_tag = node.args[0].fun.args[0]  # type: ignore[attr-defined]
            if ignored_tag == shift_tag:
                return node.args[0]

            shift = node.fun
            it = node.args[0].args  # type: ignore[attr-defined]
            ignore_shift = node.args[0].fun  # type: ignore[attr-defined]

            return ir.FunCall(fun=ignore_shift, args=[self.visit(ir.FunCall(fun=shift, args=it))])

        return node
