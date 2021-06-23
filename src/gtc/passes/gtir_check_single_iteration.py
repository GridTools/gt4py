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

import eve
from gtc import gtir


class CheckSingleIteration(eve.NodeVisitor):
    def visit_For(self, node: gtir.For, **kwargs: Any) -> None:
        offcenter_accesses = (
            node.iter_tree()
            .if_isinstance(gtir.FieldAccess)
            .filter(lambda acc: acc.offset.i != 0 or acc.offset.j != 0)
            .getattr("name")
            .to_set()
        )

        writes = (
            node.iter_tree()
            .if_isinstance(gtir.ParAssignStmt)
            .getattr("left")
            .getattr("name")
            .to_set()
        )

        offcenter_reads_and_write = offcenter_accesses & writes
        if offcenter_reads_and_write:
            raise ValueError(
                f"Detected fields that are both read with offset and written to inside a for loop. Fields: {', '.join(offcenter_reads_and_write)}"
            )


def check_single_iteration(gtir: gtir.Stencil) -> gtir.Stencil:
    """Check if there's more than one horizontal execution in a for loop."""
    CheckSingleIteration().visit(gtir)
    return gtir
