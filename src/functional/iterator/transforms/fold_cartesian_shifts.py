from eve import NodeTranslator
from functional.iterator import ir


class FoldCartesianShifts(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "shift":
            # TODO I thought we cannot reorder, therefore this implementation.
            # But actually we can, as we can apply this only to Cartesian anyway...
            # shift(o1, 1, o2, 1, o1, 1)(it) -> shift(o1,2, o2,1)(it)
            new_offsets = []
            last_inserted_tag = None
            accumulate = False
            for o in node.args:
                if isinstance(o, ir.OffsetLiteral) and isinstance(o.value, str):
                    if last_inserted_tag is not None and last_inserted_tag == o:
                        accumulate = True
                    else:
                        accumulate = False
                        new_offsets.append(o)
                        last_inserted_tag = o
                elif isinstance(o, ir.OffsetLiteral) and isinstance(o.value, int):
                    if accumulate:
                        assert isinstance(new_offsets[-1].value, int)
                        new_value = new_offsets[-1].value + o.value
                        if new_value == 0:
                            new_offsets.pop()
                            new_offsets.pop()
                            last_inserted_tag = None
                        else:
                            new_offsets[-1] = ir.OffsetLiteral(value=new_value)
                        accumulate = False
                    else:
                        new_offsets.append(o)
                    accumulate = False
                else:
                    new_offsets.append(o)
                    accumulate = False
                    last_inserted_tag = None

            return ir.FunCall(fun=ir.SymRef(id="shift"), args=new_offsets)

        return node
