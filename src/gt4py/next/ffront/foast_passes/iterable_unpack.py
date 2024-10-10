# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.next.ffront.foast_passes.utils import compute_assign_indices


class UnpackedAssignPass(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Explicitly unpack assignments.

    Example
    -------
    # before pass
    def foo():
         a0 = 1
         b0 = 2
         a1, b1 = b0, a0
         return a1, b1

    # after pass
    def foo():
        a0, b0 = (1, 2)
        __tuple_tmp_0 = (1, 2)
        a1 = __tuple_tmp_0[0]
        b1 = __tuple_tmp_0[1]
        return (a1, b1)
    """

    unique_tuple_symbol_id: int = 0

    @classmethod
    def apply(cls, node: foast.FunctionDefinition) -> foast.FunctionDefinition:
        node = cls().visit(node)
        return node

    def _unique_tuple_symbol(self, node: foast.TupleTargetAssign) -> foast.Symbol[Any]:
        sym: foast.Symbol = foast.Symbol(
            id=f"__tuple_tmp_{self.unique_tuple_symbol_id}",
            type=node.value.type,
            location=node.location,
        )
        self.unique_tuple_symbol_id += 1
        return sym

    def visit_BlockStmt(self, node: foast.BlockStmt, **kwargs: Any) -> foast.BlockStmt:
        unrolled_stmts: list[foast.Assign | foast.BlockStmt | foast.Return] = []

        for stmt in node.stmts:
            if isinstance(stmt, foast.TupleTargetAssign):
                num_elts, targets = len(stmt.value.type.types), stmt.targets  # type: ignore
                indices = compute_assign_indices(targets, num_elts)
                tuple_symbol = self._unique_tuple_symbol(stmt)
                unrolled_stmts.append(
                    foast.Assign(target=tuple_symbol, value=stmt.value, location=stmt.location)
                )

                for index, subtarget in zip(indices, targets):
                    el_type = subtarget.type
                    tuple_name = foast.Name(
                        id=tuple_symbol.id, type=el_type, location=tuple_symbol.location
                    )
                    if isinstance(index, tuple):  # handle starred target
                        lower, upper = index
                        slice_indices = list(range(lower, upper))
                        tuple_slice = [
                            foast.Subscript(
                                value=tuple_name, index=i, type=el_type, location=stmt.location
                            )
                            for i in slice_indices
                        ]

                        new_tuple = foast.TupleExpr(
                            elts=tuple_slice, type=el_type, location=stmt.location
                        )
                        new_assign = foast.Assign(
                            target=subtarget.id, value=new_tuple, location=stmt.location
                        )
                    else:
                        new_assign = foast.Assign(
                            target=subtarget,
                            value=foast.Subscript(
                                value=tuple_name, index=index, type=el_type, location=stmt.location
                            ),
                            location=stmt.location,
                        )
                    unrolled_stmts.append(new_assign)
            else:
                unrolled_stmts.append(self.generic_visit(stmt, **kwargs))

        return foast.BlockStmt(stmts=unrolled_stmts, location=node.location)
