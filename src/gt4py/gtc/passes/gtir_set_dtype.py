from eve import NodeTranslator

from gt4py.gtc import gtir


# TODO How do we deal with AUTO types?
# Especially in the dtype propagator, currently AUTO is included in strict type checking.


class GTIRSetDtype(NodeTranslator):
    def visit_FieldAccess(self, node: gtir.FieldAccess, *, symtable, **kwargs):
        return gtir.FieldAccess(
            name=node.name, offset=node.offset, dtype=symtable[node.name].dtype
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return self.generic_visit(node, symtable=node.symtable_)
