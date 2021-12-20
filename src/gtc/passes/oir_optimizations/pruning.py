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

from typing import Any

from eve import NOTHING, NodeTranslator
from gtc import oir


class NoFieldAccessPruning(NodeTranslator):
    def visit_HorizontalExecution(self, node: oir.HorizontalExecution) -> Any:
        try:
            next(iter(node.iter_tree().if_isinstance(oir.FieldAccess)))
        except StopIteration:
            return NOTHING
        return node

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection) -> Any:
        horizontal_executions = self.visit(node.horizontal_executions)
        if not horizontal_executions:
            return NOTHING
        return oir.VerticalLoopSection(
            interval=node.interval,
            horizontal_executions=horizontal_executions,
            loc=node.loc,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop) -> Any:
        sections = self.visit(node.sections)
        if not sections:
            return NOTHING
        return oir.VerticalLoop(
            loop_order=node.loop_order,
            sections=sections,
            caches=node.caches,
            loc=node.loc,
        )
