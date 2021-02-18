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

from typing import Any, Dict, Set, Union

import eve
from gtc import common, oir
from gtc.cuir import cuir


class OIRToCUIR(eve.NodeTranslator):
    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> cuir.Literal:
        return cuir.Literal(value=node.value, dtype=node.dtype)

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.FieldDecl(name=node.name, dtype=node.dtype)

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> cuir.FieldDecl:
        return cuir.ScalarDecl(name=node.name, dtype=node.dtype)

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> cuir.UnaryOp:
        return cuir.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs), dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> cuir.BinaryOp:
        return cuir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
            dtype=node.dtype,
        )

    def visit_Temporary(self, node: oir.Temporary, **kwargs: Any) -> cuir.Temporary:
        return cuir.Temporary(name=node.name, dtype=node.dtype)

    def visit_FieldAccess(
        self, node: oir.FieldAccess, *, ij_cached: Set[str], k_cached: Set[str], **kwargs: Any
    ) -> Union[cuir.FieldAccess, cuir.IJCacheAccess, cuir.KCacheAccess]:
        if node.name in ij_cached:
            assert node.offset.k == 0
            return cuir.IJCacheAccess(
                name=node.name, offset=(node.offset.i, node.offset.j), dtype=node.dtype
            )
        if node.name in k_cached:
            assert node.offset.i == node.offset.j == 0
            return cuir.KCacheAccess(name=node.name, offset=node.offset.k, dtype=node.dtype)
        return cuir.FieldAccess(name=node.name, offset=node.offset, dtype=node.dtype)

    def visit_ScalarAccess(
        self, node: oir.ScalarAccess, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> cuir.ScalarAccess:
        if isinstance(symtable.get(node.name, None), oir.ScalarDecl):
            return cuir.FieldAccess(
                name=node.name, offset=common.CartesianOffset.zero(), dtype=node.dtype
            )
        return cuir.ScalarAccess(name=node.name, dtype=node.dtype)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> cuir.AssignStmt:
        return cuir.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> cuir.Cast:
        return cuir.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> cuir.NativeFuncCall:
        return cuir.NativeFuncCall(
            func=node.func, args=self.visit(node.args, **kwargs), dtype=node.dtype
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> cuir.TernaryOp:
        return cuir.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
            dtype=node.dtype,
        )

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, **kwargs: Any
    ) -> cuir.HorizontalExecution:
        return cuir.HorizontalExecution(
            body=self.visit(node.body, **kwargs),
            mask=self.visit(node.mask, **kwargs),
            declarations=self.visit(node.declarations),
        )

    def visit_VerticalLoopSection(
        self, node: oir.VerticalLoopSection, **kwargs: Any
    ) -> cuir.VerticalLoopSection:
        return cuir.VerticalLoopSection(
            start=node.interval.start,
            end=node.interval.end,
            horizontal_executions=self.visit(node.horizontal_executions, **kwargs),
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> cuir.Kernel:
        assert not any(c.fill or c.flush for c in node.caches if isinstance(c, oir.KCache))
        ij_cached = {c.name for c in node.caches if isinstance(c, oir.IJCache)}
        k_cached = {c.name for c in node.caches if isinstance(c, oir.KCache)}
        return cuir.Kernel(
            name=node.id_,
            vertical_loops=[
                cuir.VerticalLoop(
                    loop_order=node.loop_order,
                    sections=self.visit(
                        node.sections, ij_cached=ij_cached, k_cached=k_cached, **kwargs
                    ),
                )
            ],
        )

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> cuir.Program:
        return cuir.Program(
            name=node.name,
            params=self.visit(node.params),
            declarations=self.visit(node.declarations),
            kernels=self.visit(node.vertical_loops, symtable=node.symtable_),
        )
