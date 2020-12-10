from eve import NodeTranslator

from gt4py.gtc import gtir
from gt4py.gtc.common import DataType

from devtools import debug


class _GTIRResolveAuto(NodeTranslator):
    """
    Replaces AUTO dtype by a concrete dtype.

    Precondition: All dtype are set (not None)
    Postcondition: All dtypes are concrete (no AUTO)
    """

    class _GTIRUpdateAutoDecl(NodeTranslator):
        """Updates FieldDecls with resolved types"""

        def visit_FieldDecl(self, node: gtir.FieldDecl, new_symbols, **kwargs):
            if node.dtype == DataType.AUTO:
                dtype = new_symbols[node.name].dtype
                return gtir.FieldDecl(name=node.name, dtype=dtype)
            else:
                return node

    def visit_FieldAccess(self, node: gtir.FieldAccess, *, symtable, **kwargs):
        if symtable[node.name].dtype == DataType.AUTO:
            assert "new_dtype" in kwargs
            symtable[node.name].dtype = kwargs["new_dtype"]
        return gtir.FieldAccess(
            name=node.name, offset=node.offset, dtype=symtable[node.name].dtype
        )

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs):
        right = self.visit(node.right, **kwargs)
        left = self.visit(node.left, new_dtype=right.dtype, **kwargs)
        return gtir.ParAssignStmt(left=left, right=right)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        symtable = node.symtable_
        result = self.generic_visit(node, symtable=symtable)
        result = self._GTIRUpdateAutoDecl().visit(result, new_symbols=symtable)

        # TODO enable after FieldsMetaData is updated
        # assert all(
        #     result.iter_tree()
        #     .if_hasattr("dtype")
        #     .getattr("dtype")
        #     .map(lambda x: x not in [None, DataType.AUTO, DataType.INVALID, DataType.DEFAULT])
        # )
        return result


class _GTIRPropagateDtypeToAccess(NodeTranslator):
    """
    Propagates dtype from Decl to Access

    Precondition: Decls have dtype (not None), can be AUTO or DEFAULT
    Postcondition: All dtypes of Access are not None
    """

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
    return _GTIRResolveAuto().visit(_GTIRPropagateDtypeToAccess().visit(node))
