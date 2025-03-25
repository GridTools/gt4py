from typing import Any

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im


class BindArgs(PreserveLocationVisitor, NodeTranslator):
    def visit_SetAt(self, node: ir.SetAt):
        assert isinstance(node.expr, ir.FunCall)
        new_args = [
            im.literal_from_value(self.bound_args[arg.id])
            if (isinstance(arg, ir.SymRef) and arg.id in self.bound_args)
            else arg
            for arg in node.expr.args
        ]
        return ir.SetAt(
            expr=ir.FunCall(
                fun=node.expr.fun,
                args=new_args,
            ),
            domain=node.domain,
            target=node.target,
        )

    @classmethod
    def apply(cls, node: ir.Node, **bound_args: Any) -> ir.Node:
        obj = cls()
        obj.bound_args = bound_args
        return obj.visit(node)
