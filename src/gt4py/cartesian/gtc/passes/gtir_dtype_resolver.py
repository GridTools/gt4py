# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict

from gt4py import eve
from gt4py.cartesian.gtc import gtir
from gt4py.cartesian.gtc.common import DataType, GTCPostconditionError


class _GTIRResolveAuto(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    """
    Replaces AUTO dtype by a concrete dtype.

    Note that currently only temporaries (FieldDecl/FieldAccess) can have AUTO type.

    Precondition: All dtype are set, but some can be set to AUTO
    Postcondition: All dtypes are concrete (no AUTO)
    """

    class _GTIRUpdateAutoDecl(eve.NodeTranslator):
        """Updates FieldDecls with resolved types."""

        def visit_FieldDecl(
            self, node: gtir.FieldDecl, new_symbols: Dict[str, Any], **kwargs: Any
        ) -> gtir.FieldDecl:
            if node.dtype == DataType.AUTO:
                dtype = new_symbols[node.name].dtype
                return gtir.FieldDecl(
                    name=node.name, dtype=dtype, dimensions=node.dimensions, loc=node.loc
                )
            else:
                return node

    def visit_FieldAccess(
        self, node: gtir.FieldAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> gtir.FieldAccess:
        if symtable[node.name].dtype == DataType.AUTO:
            assert "new_dtype" in kwargs
            symtable[node.name].dtype = kwargs["new_dtype"]
        return gtir.FieldAccess(
            name=node.name,
            offset=self.visit(node.offset, symtable=symtable, **kwargs),
            data_index=self.visit(node.data_index, symtable=symtable, **kwargs),
            dtype=symtable[node.name].dtype,
            loc=node.loc,
        )

    def visit_IteratorAccess(self, iterator_access: gtir.IteratorAccess, **kwargs: Any):
        return gtir.IteratorAccess(name=iterator_access.name, dtype=DataType.INT64)

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs: Any) -> gtir.ParAssignStmt:
        right = self.visit(node.right, **kwargs)
        left = self.visit(node.left, new_dtype=right.dtype, **kwargs)
        return gtir.ParAssignStmt(left=left, right=right, loc=node.loc)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> gtir.Stencil:
        result = self.generic_visit(node, **kwargs)
        result = self._GTIRUpdateAutoDecl().visit(result, new_symbols=kwargs["symtable"])

        if not all(
            result.walk_values()
            .if_hasattr("dtype")
            .getattr("dtype")
            .map(lambda x: x not in {DataType.AUTO, DataType.INVALID, DataType.DEFAULT})
        ):
            raise GTCPostconditionError(expected="No AUTO, INVALID or DEFAULT dtype in tree.")

        return result


class _GTIRPropagateDtypeToAccess(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    """
    Propagates dtype from Decl to Access.

    Precondition: Decls have dtype (not None), can be AUTO or DEFAULT
    Postcondition: All dtypes of Access are not None
    """

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs: Any) -> gtir.FieldAccess:
        return gtir.FieldAccess(
            name=node.name,
            offset=self.visit(node.offset, **kwargs),
            data_index=self.visit(node.data_index, **kwargs),
            dtype=kwargs["symtable"][node.name].dtype,
            loc=node.loc,
        )

    def visit_ScalarAccess(
        self, node: gtir.ScalarAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> gtir.ScalarAccess:
        return gtir.ScalarAccess(name=node.name, dtype=symtable[node.name].dtype, loc=node.loc)


def resolve_dtype(node: gtir.Stencil) -> gtir.Stencil:
    return _GTIRResolveAuto().visit(_GTIRPropagateDtypeToAccess().visit(node))
