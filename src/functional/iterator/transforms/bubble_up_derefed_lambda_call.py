from eve import NodeTranslator
from functional.iterator import ir


class BubbleUpDerefedLambdaCall(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)  # depth-first
        if node.fun == ir.SymRef(id="deref"):
            if isinstance((fun_call := node.args[0]), ir.FunCall) and isinstance(
                (lambda_ := fun_call.fun), ir.Lambda
            ):
                new_lambda = ir.Lambda(
                    params=lambda_.params,
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[lambda_.expr]),
                )
                return ir.FunCall(fun=new_lambda, args=fun_call.args)
        return node
