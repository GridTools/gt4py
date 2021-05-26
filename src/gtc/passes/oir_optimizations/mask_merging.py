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

from typing import Any, List, Optional

from eve import NodeTranslator
from gtc import oir


class MaskMerging(NodeTranslator):
    """Merges consecutive mask statements if the conditionals are the same."""

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, **kwargs: Any
    ) -> oir.HorizontalExecution:
        merged_body: List[oir.Stmt] = []
        mask_statement: Optional[oir.MaskStmt] = None

        for statement in node.body:
            if isinstance(statement, oir.MaskStmt):
                if mask_statement:
                    if statement.mask == mask_statement.mask:
                        mask_statement.body += statement.body
                        continue
                else:
                    mask_statement = statement
            merged_body.append(statement)

        return oir.HorizontalExecution(
            body=merged_body,
            declarations=node.declarations,
        )
