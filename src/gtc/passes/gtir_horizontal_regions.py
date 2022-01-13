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

from typing import Set

import eve
from gtc import gtir


class CheckHorizontalRegionAccesses(eve.NodeVisitor):
    """Ensure that FieldAccess nodes in HorizontalRegions access up-to-date memory."""

    def visit_VerticalLoop(self, node: gtir.VerticalLoop) -> None:
        self.visit(node.body, fields_set=set())

    def visit_HorizontalRegion(self, node: gtir.HorizontalRegion, *, fields_set: Set[str]) -> None:
        self.visit(node.block, fields_set=fields_set, inside_region=True)

    def visit_ParAssignStmt(
        self, node: gtir.FieldAccess, *, fields_set: Set[str], **kwargs
    ) -> None:
        self.visit(node.right, fields_set=fields_set, **kwargs)
        fields_set.add(node.left.name)

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        fields_set: Set[str],
        inside_region: bool = False,
    ) -> None:
        offsets = node.offset.to_dict()
        zero_horizontal_offset = offsets["i"] == 0 and offsets["j"] == 0
        if inside_region and not zero_horizontal_offset and node.name in fields_set:
            # This access will potentially read memory that has not been updated yet
            raise ValueError(f"Race condition detected on read of {node.name}")


def check_horizontal_regions(stencil: gtir.Stencil) -> gtir.Stencil:
    CheckHorizontalRegionAccesses().visit(stencil)
    return stencil
