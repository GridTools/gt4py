from eve import NodeTranslator

from gt4py.gtc import gtir
from gt4py.gtc.common import DataType, ScalarAccess

from devtools import debug

# TODO How do we deal with AUTO types?
# Especially in the dtype propagator, currently AUTO is included in strict type checking.


class GTIRUpdateAutoDecl(NodeTranslator):
    def visit_FieldDecl(self, node: gtir.FieldDecl, new_symbols, **kwargs):
        dtype = new_symbols[node.name].dtype
        return gtir.FieldDecl(name=node.name, dtype=dtype)


class GTIRResolveAuto(NodeTranslator):
    def visit_FieldAccess(self, node: gtir.FieldAccess, *, symtable, **kwargs):
        if symtable[node.name].dtype == DataType.AUTO:
            assert "new_dtype" in kwargs
            symtable[node.name].dtype = kwargs["new_dtype"]
        return gtir.FieldAccess(
            name=node.name, offset=node.offset, dtype=symtable[node.name].dtype
        )

        # if new_dtype and symtable[node.name].dtype == DataType.AUTO:
        #     symtable[node.name].dtype = new_dtype

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs):
        right = self.visit(node.right, **kwargs)
        left = self.visit(node.left, new_dtype=right.dtype, **kwargs)
        return gtir.ParAssignStmt(left=left, right=right)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        symtable = node.symtable_
        result = self.generic_visit(node, symtable=symtable)
        result = GTIRUpdateAutoDecl().visit(result, new_symbols=symtable)  # quite a hack
        return result
        # symtable = node.symtable_
        # debug(symtable)
        # return result
        # return self.generic_visit(node, symtable=node.symtable_)


class GTIRSetDtype(NodeTranslator):
    def visit_FieldAccess(self, node: gtir.FieldAccess, *, symtable, **kwargs):
        return gtir.FieldAccess(
            name=node.name, offset=node.offset, dtype=symtable[node.name].dtype
        )

    def visit_ScalarAccess(self, node: gtir.ScalarAccess, *, symtable, **kwargs):
        return gtir.ScalarAccess(name=node.name, dtype=symtable[node.name].dtype)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        result: gtir.Stencil = self.generic_visit(node, symtable=node.symtable_)
        assert all(
            result.iter_tree()
            .if_isinstance(gtir.ScalarAccess, gtir.FieldAccess)
            .getattr("dtype")
            .map(lambda x: x is not None)
        )
        return result


def resolve_dtype(node: gtir.Stencil):
    return GTIRResolveAuto().visit(GTIRSetDtype().visit(node))
