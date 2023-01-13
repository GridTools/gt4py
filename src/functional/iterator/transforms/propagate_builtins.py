from eve import NodeTranslator
from eve.pattern_matching import ObjectPattern as P
from functional.iterator import ir


class PropagateBuiltins(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        """
        Propagate calls to builtins into lambda functions.

        Transform::

            builtin(λ(inner_arg) → inner_arg)(outer_arg)

        into::

            (λ(inner_arg) → builtin(inner_arg))(outer_arg)

        After this pass calls to builtins are grouped together (better) with lambda functions
        being propagated outwards. This increases the ability of other passes to transform /
        optimize builtins. For example `deref` calls, e.g.::

            ·(λ(inner_it) → lift(stencil)(inner_it))(outer_it)

        are propagated inwards::

            (λ(inner_it) → ·(lift(stencil)(inner_it)))(outer_it)

        after which the lift inliner can remove the deref + lift combination.
        """
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        if P(ir.FunCall, fun=P(ir.SymRef), args=[P(ir.FunCall, fun=P(ir.Lambda))]).match(node):
            builtin = node.fun
            lambda_fun: ir.Lambda = node.args[0].fun  # type: ignore[attr-defined]
            lambda_args: list[ir.Expr] = node.args[0].args  # type: ignore[attr-defined]
            node = ir.FunCall(
                fun=ir.Lambda(
                    params=lambda_fun.params,
                    expr=ir.FunCall(fun=builtin, args=[lambda_fun.expr]),
                ),
                args=lambda_args,
            )
        return self.generic_visit(node)
