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

from eve import NodeTranslator
from gtc import oir


class ZeroExtentMerging(NodeTranslator):
    """Merges horizontal executions with zero offsets with previous horizontal exectutions within the same vertical loop."""

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        assert node.horizontal_executions, "non-empty vertical loop expected"
        result = self.generic_visit(node)
        horizontal_executions = [result.horizontal_executions[0]]
        for horizontal_execution in result.horizontal_executions[1:]:
            has_zero_extents = (
                horizontal_execution.iter_tree()
                .if_isinstance(oir.FieldAccess)
                .map(lambda x: x.offset.i == x.offset.j == 0)
                .reduce(lambda a, b: a and b, init=True)
            )
            # TODO: support masks?
            if has_zero_extents and horizontal_execution.mask is None:
                horizontal_executions[-1].body += horizontal_execution.body
            else:
                horizontal_executions.append(horizontal_execution)
        result.horizontal_executions = horizontal_executions
        return result
