from eve import NodeTranslator
from eve.pattern_matching import ObjectPattern as P
from functional.iterator import ir


class InlineTupleGet(NodeTranslator):
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        if P(ir.FunCall, fun=ir.SymRef(id="tuple_get")).match(node):
            assert len(node.args) == 2
            index, tuple_ = node.args
            if P(ir.FunCall, fun=ir.SymRef(id="make_tuple")).match(tuple_):
                assert isinstance(index, ir.Literal) and index.type == "int"
                return self.generic_visit(tuple_.args[int(index.value)])  # type: ignore[attr-defined]
        return self.generic_visit(node)
