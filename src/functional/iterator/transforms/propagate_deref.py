from eve import NodeTranslator
from eve.pattern_matching import ObjectPattern as P
from functional.iterator import ir


class PropagateDeref(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        # Transform
        #  ·(λ(inner_arg) → inner_arg)(outer_arg)
        # into
        #  (λ(inner_arg) → ·inner_arg)(outer_arg)
        if P(ir.FunCall, fun=ir.SymRef(id="deref"), args=[P(ir.FunCall, fun=P(ir.Lambda))]).match(
            node
        ):
            lambda_fun: ir.Lambda = node.args[0].fun
            lambda_args: list[ir.Expr] = node.args[0].args
            node = ir.FunCall(
                fun=ir.Lambda(
                    params=lambda_fun.params,
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[lambda_fun.expr]),
                ),
                args=lambda_args,
            )
        return self.generic_visit(node)
