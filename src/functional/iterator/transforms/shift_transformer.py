from eve import NOTHING, NodeTranslator
from eve.pattern_matching import ObjectPattern as P
from functional.iterator import ir


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
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"), args=[node.args[0].args[0]]))

        # deref(translate_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))],
        ).match(node):
            return self.visit(ir.FunCall(fun=ir.SymRef(id="deref"), args=[node.args[0].args[0]]))

        # can_deref(ignore_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="can_deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))],
        ).match(node):
            return self.visit(
                ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[node.args[0].args[0]])
            )

        # can_deref(translate_shift(...)(it)) -> deref(it)
        if P(
            ir.FunCall,
            fun=ir.SymRef(id="can_deref"),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="translate_shift")))],
        ).match(node):
            return self.visit(
                ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[node.args[0].args[0]])
            )

        return self.generic_visit(node)


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
            assert len(node.fun.args) == 2
            shift_tag, shift_index = node.fun.args
            old_tag, new_tag = node.args[0].fun.args
            if old_tag == shift_tag:
                shift_tag = new_tag

            new_shift = ir.FunCall(fun=ir.SymRef(id="shift"), args=[shift_tag, shift_index])
            translate_shift = node.args[0].fun
            it = node.args[0].args

            return ir.FunCall(
                fun=translate_shift, args=[self.visit(ir.FunCall(fun=new_shift, args=it))]
            )
        # shift(...)(ignore_shift(...)(it)) -> ignore_shift(...)(shift(...)(it))
        elif P(
            ir.FunCall,
            fun=P(ir.FunCall, fun=ir.SymRef(id="shift")),
            args=[P(ir.FunCall, fun=P(ir.FunCall, fun=ir.SymRef(id="ignore_shift")))],
        ).match(node):
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
