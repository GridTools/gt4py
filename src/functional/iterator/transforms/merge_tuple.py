import eve
from functional.iterator import ir


class MergeTuple(eve.NodeTranslator):
    """Transform `make_tuple(tuple_get(0, t), tuple_get(1, t), ...)` -> t."""

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if node.fun == ir.SymRef(id="make_tuple") and all(
            isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
            for arg in node.args
        ):
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]
            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and v.args[1] == first_expr):
                    return self.generic_visit(node)

            return first_expr
        return self.generic_visit(node)
