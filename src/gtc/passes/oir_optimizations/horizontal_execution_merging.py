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
from gtc.common import GTCPostconditionError, GTCPreconditionError

from .utils import AccessCollector


class GreedyMerging(NodeTranslator):
    """Merges consecutive horizontal executions if there are no write/read conflicts.

    Preconditions: All vertical loops are non-empty.
    Postcondition: The number of horizontal executions is equal or smaller than before.
    """

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> oir.VerticalLoop:
        if not node.horizontal_executions:
            raise GTCPreconditionError(expected="non-empty vertical loop")
        result = self.generic_visit(node, **kwargs)
        horizontal_executions = [result.horizontal_executions[0]]
        accesses = AccessCollector.apply(horizontal_executions[-1])
        previous_reads = accesses.read_offsets()
        previous_writes = accesses.write_offsets()
        for horizontal_execution in result.horizontal_executions[1:]:
            accesses = AccessCollector.apply(horizontal_execution)
            current_reads = accesses.read_offsets()
            current_writes = accesses.write_offsets()

            conflicting = {
                field
                for field, offsets in current_reads.items()
                if field in previous_writes and offsets ^ previous_writes[field]
            } | {
                field
                for field, offsets in current_writes.items()
                if field in previous_reads
                and any(o[:2] != (0, 0) for o in offsets ^ previous_reads[field])
            }
            if not conflicting and horizontal_execution.mask == horizontal_executions[-1].mask:
                horizontal_executions[-1].body += horizontal_execution.body
                for field, writes in current_writes.items():
                    previous_writes[field] |= writes
                for field, reads in current_reads.items():
                    previous_reads[field] |= reads
            else:
                horizontal_executions.append(horizontal_execution)
                previous_writes = current_writes
                previous_reads = current_reads
        result.horizontal_executions = horizontal_executions
        if len(result.horizontal_executions) > len(node.horizontal_executions):
            raise GTCPostconditionError(
                expected="the number of horizontal executions is equal or smaller than before"
            )
        return result
