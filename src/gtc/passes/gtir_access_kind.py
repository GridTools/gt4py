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


from typing import Any, Dict, Union

from eve.visitors import NodeVisitor
from gt4py.definitions import AccessKind
from gtc import common, gtir


class AccessKindComputer(NodeVisitor):
    def _visit_Access(
        self, name, *, whole_interval: bool, access: Dict[str, AccessKind], kind: AccessKind
    ) -> None:
        if kind == AccessKind.WRITE and access.get(name, None) == AccessKind.READ:
            access[name] = AccessKind.READ_WRITE
        elif name not in access:
            access[name] = kind

    def _visit_If(self, node: Union[gtir.FieldIfStmt, gtir.ScalarIfStmt], **kwargs: Any) -> None:
        self.visit(node.cond, kind=AccessKind.READ, **kwargs)
        self.visit(node.true_branch, **kwargs)
        if node.false_branch:
            self.visit(node.false_branch, **kwargs)

    def visit_ScalarAccess(self, node: gtir.ScalarAccess, **kwargs: Any) -> None:
        self._visit_Access(node.name, **kwargs)

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs: Any) -> None:
        self.generic_visit(node, **kwargs)
        self._visit_Access(node.name, **kwargs)

    def visit_ScalarIfStmt(self, node: gtir.ScalarIfStmt, **kwargs: Any) -> None:
        self._visit_If(node, **kwargs)

    def visit_FieldIfStmt(self, node: gtir.FieldIfStmt, **kwargs: Any) -> None:
        self._visit_If(node, **kwargs)

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs: Any) -> None:
        self.visit(node.right, kind=AccessKind.READ, **kwargs)
        self.visit(node.left, kind=AccessKind.WRITE, **kwargs)

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs: Any):
        whole_interval = node.interval == gtir.Interval(
            start=common.AxisBound.start(), end=common.AxisBound.end()
        )
        self.generic_visit(node, whole_interval=whole_interval, **kwargs)

    def visit_Stencil(self, node: gtir.Stencil) -> Dict[str, AccessKind]:
        access: Dict[str, AccessKind] = {}
        self.generic_visit(node, access=access)
        for param in node.params:
            if param.name not in access:
                access[param.name] = AccessKind.NONE
        return access


def compute_access_kinds(stencil: gtir.Stencil) -> Dict[str, AccessKind]:
    return AccessKindComputer().visit(stencil)
