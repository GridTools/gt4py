from eve import NodeTranslator
from functional.iterator import ir


def _is_call_to_shift(node):
    return (
        isinstance(node, ir.FunCall) and isinstance(node.fun, ir.SymRef) and node.fun.id == "shift"
    )


class NormalizeShifts(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if (
            _is_call_to_shift(node.fun)
            and node.args
            and isinstance(node.args[0], ir.FunCall)
            and _is_call_to_shift(node.args[0].fun)
        ):
            # shift(args1...)(shift(args2...)(it)) -> shift(args2..., args1...)(it)
            assert len(node.args) == 1
            return ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"), args=node.args[0].fun.args + node.fun.args
                ),
                args=node.args[0].args,
            )
        return node
