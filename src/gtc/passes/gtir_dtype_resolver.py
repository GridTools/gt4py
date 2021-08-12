# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict

from eve import NodeTranslator, SymbolTableTrait
from gtc import gtir
from gtc.common import DataType, GTCPostconditionError


class _GTIRResolveAuto(NodeTranslator):
    """
    Replaces AUTO dtype by a concrete dtype.

    Note that currently only temporaries (FieldDecl/FieldAccess) can have AUTO type.

    Precondition: All dtype are set (not None)
    Postcondition: All dtypes are concrete (no AUTO)
    """

    contexts = (SymbolTableTrait.symtable_merger,)

    class _GTIRUpdateAutoDecl(NodeTranslator):
        """Updates FieldDecls with resolved types."""

        def visit_FieldDecl(
            self, node: gtir.FieldDecl, new_symbols: Dict[str, Any], **kwargs: Any
        ) -> gtir.FieldDecl:
            if node.dtype == DataType.AUTO:
                dtype = new_symbols[node.name].dtype
                return gtir.FieldDecl(name=node.name, dtype=dtype, dimensions=node.dimensions)
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
            offset=node.offset,
            data_index=node.data_index,
            dtype=symtable[node.name].dtype,
        )

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs: Any) -> gtir.ParAssignStmt:
        right = self.visit(node.right, **kwargs)
        left = self.visit(node.left, new_dtype=right.dtype, **kwargs)
        return gtir.ParAssignStmt(left=left, right=right)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> gtir.Stencil:
        result = self.generic_visit(node, **kwargs)
        result = self._GTIRUpdateAutoDecl().visit(result, new_symbols=kwargs["symtable"])

        if not all(
            result.iter_tree()
            .if_hasattr("dtype")
            .getattr("dtype")
            .map(lambda x: x not in [None, DataType.AUTO, DataType.INVALID, DataType.DEFAULT])
        ):
            raise GTCPostconditionError(expected="No AUTO, INVALID or DEFAULT dtype in tree.")

        return result


class _GTIRPropagateDtypeToAccess(NodeTranslator):
    """
    Propagates dtype from Decl to Access.

    Precondition: Decls have dtype (not None), can be AUTO or DEFAULT
    Postcondition: All dtypes of Access are not None
    """

    contexts = (SymbolTableTrait.symtable_merger,)

    def visit_FieldAccess(
        self, node: gtir.FieldAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> gtir.FieldAccess:
        return gtir.FieldAccess(
            name=node.name,
            offset=self.visit(node.offset, symtable=symtable),
            data_index=node.data_index,
            dtype=symtable[node.name].dtype,
        )

    def visit_VariableOffset(
        self, node: gtir.VariableOffset, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> gtir.VariableOffset:
        return gtir.VariableOffset(i=node.i, j=node.j, k=self.visit(node.k, symtable=symtable))

    def visit_ScalarAccess(
        self, node: gtir.ScalarAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> gtir.ScalarAccess:
        return gtir.ScalarAccess(name=node.name, dtype=symtable[node.name].dtype)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> gtir.Stencil:
        result: gtir.Stencil = self.generic_visit(node, **kwargs)

        if not all(
            result.iter_tree()
            .if_isinstance(gtir.ScalarAccess, gtir.FieldAccess)
            .getattr("dtype")
            .map(lambda x: x is not None)
        ):
            raise GTCPostconditionError(expected="No None dtype in FieldAccess or ScalarAccess.")
        return result


def resolve_dtype(node: gtir.Stencil) -> gtir.Stencil:
    return _GTIRResolveAuto().visit(_GTIRPropagateDtypeToAccess().visit(node))
