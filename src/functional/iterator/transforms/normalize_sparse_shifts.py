from eve import NodeTranslator
from functional.iterator import ir


class NormalizeSparseShifts(NodeTranslator):  # TODO should it be last?
    """Puts the sparse shift first.

    Preconditions: Shifts are normalizes (see NormalizeShifts)
    """

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "shift"
        ):
            # shift(offsets..., extra_shift_by_number, more_offsets...)(it)
            # -> shift(extra_shift_by_number, offsets..., more_offsets...)`
            offsets = []
            sparse_offsets = []
            tag_stack = []
            for arg in node.fun.args:
                assert isinstance(arg, ir.OffsetLiteral)
                if isinstance(arg.value, str):
                    tag_stack.append(arg)
                else:
                    assert isinstance(arg.value, int)
                    if tag_stack:
                        offsets.append(tag_stack.pop())
                        offsets.append(arg)
                    else:
                        sparse_offsets.append(arg)

            return ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=sparse_offsets + offsets),
                args=node.args,
            )
        return node
