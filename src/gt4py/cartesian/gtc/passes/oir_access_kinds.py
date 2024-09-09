# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
from typing import Any, Dict

from gt4py.cartesian.definitions import AccessKind
from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.passes.horizontal_masks import mask_overlap_with_extent
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import compute_horizontal_block_extents
from gt4py.eve.visitors import NodeVisitor


class AccessKindComputer(NodeVisitor):
    def _visit_Access(
        self, name, *, access: Dict[str, AccessKind], kind: AccessKind, **kwargs: Any
    ) -> None:
        if kind == AccessKind.WRITE and access.get(name, None) == AccessKind.READ:
            access[name] = AccessKind.READ_WRITE
        elif name not in access:
            access[name] = kind

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **kwargs: Any) -> None:
        self._visit_Access(node.name, **kwargs)

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> None:
        self.visit(node.offset, **{**kwargs, "kind": AccessKind.READ})
        self.visit(node.data_index, **{**kwargs, "kind": AccessKind.READ})
        self._visit_Access(node.name, **kwargs)

    def visit_While(self, node: oir.While, **kwargs: Any) -> None:
        self.visit(node.cond, kind=AccessKind.READ, **kwargs)
        self.visit(node.body, **kwargs)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> None:
        self.visit(node.right, kind=AccessKind.READ, **kwargs)
        self.visit(node.left, kind=AccessKind.WRITE, **kwargs)

    def visit_HorizontalRestriction(
        self, node: oir.HorizontalRestriction, *, horizontal_extent: Extent, **kwargs: Any
    ) -> None:
        if mask_overlap_with_extent(node.mask, horizontal_extent):
            self.visit(node.mask, kind=AccessKind.READ, **kwargs)
            self.visit(node.body, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> None:
        self.visit(node.mask, kind=AccessKind.READ, **kwargs)
        self.visit(node.body, **kwargs)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, **kwargs: Any) -> None:
        self.generic_visit(node, horizontal_extent=kwargs["block_extents"][id(node)], **kwargs)

    def visit_Stencil(self, node: oir.Stencil) -> Dict[str, AccessKind]:
        access: Dict[str, AccessKind] = collections.defaultdict(lambda: AccessKind.NONE)
        block_extents = compute_horizontal_block_extents(node)
        self.generic_visit(node, access=access, block_extents=block_extents)
        return access


def compute_access_kinds(stencil: oir.Stencil) -> Dict[str, AccessKind]:
    return AccessKindComputer().visit(stencil)
