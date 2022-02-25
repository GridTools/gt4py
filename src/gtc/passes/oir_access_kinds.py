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


import collections
from typing import Any, Dict

from eve.visitors import NodeVisitor
from gt4py.definitions import AccessKind, Extent
from gtc import oir
from gtc.passes.oir_optimizations.utils import GeneralAccess, compute_horizontal_block_extents


def horizontal_mask_overlaps_block_extent(mask: oir.HorizontalMask, block_extent: Extent) -> bool:
    return all(
        GeneralAccess._overlap_along_axis(axis_extent, interval) is not None
        for axis_extent, interval in zip(block_extent, mask.intervals)
    )


class AccessKindComputer(NodeVisitor):
    def _visit_Access(
        self, name, *, access: Dict[str, AccessKind], kind: AccessKind, **kwargs: Any
    ) -> None:
        if kind == AccessKind.WRITE and access.get(name, None) == AccessKind.READ:
            access[name] = AccessKind.READ_WRITE
        elif name not in access:
            access[name] = kind

    def _visit_Mask(self, node: oir.MaskStmt, **kwargs: Any) -> None:
        self.visit(node.mask, kind=AccessKind.READ, **kwargs)
        self.visit(node.true_branch, **kwargs)
        if node.false_branch:
            self.visit(node.false_branch, **kwargs)

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **kwargs: Any) -> None:
        self._visit_Access(node.name, **kwargs)

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs: Any) -> None:
        self.generic_visit(node, **kwargs)
        self._visit_Access(node.name, **kwargs)

    def visit_While(self, node: oir.While, **kwargs: Any) -> None:
        self.visit(node.cond, kind=AccessKind.READ, **kwargs)
        self.visit(node.body, **kwargs)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs: Any) -> None:
        self.visit(node.right, kind=AccessKind.READ, **kwargs)
        self.visit(node.left, kind=AccessKind.WRITE, **kwargs)

    def visit_MaskStmt(
        self, node: oir.MaskStmt, *, horizontal_extent: Extent, **kwargs: Any
    ) -> None:
        if masks := node.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list():
            # Masks cannot be nested, so this is always valid.
            mask = masks[0]
        else:
            mask = None

        if (mask and horizontal_mask_overlaps_block_extent(mask, horizontal_extent)) or not mask:
            # Could pass horizontal_extent for more analysis capability.
            self._visit_Mask(node, **kwargs)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, **kwargs: Any) -> None:
        self.generic_visit(node, horizontal_extent=kwargs["block_extents"][id(node)], **kwargs)

    def visit_Stencil(self, node: oir.Stencil) -> Dict[str, AccessKind]:
        access: Dict[str, AccessKind] = collections.defaultdict(lambda: AccessKind.NONE)
        block_extents = compute_horizontal_block_extents(node)
        self.generic_visit(node, access=access, block_extents=block_extents)
        return access


def compute_access_kinds(stencil: oir.Stencil) -> Dict[str, AccessKind]:
    return AccessKindComputer().visit(stencil)
