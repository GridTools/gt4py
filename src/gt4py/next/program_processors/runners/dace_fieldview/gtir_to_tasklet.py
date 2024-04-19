# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


from typing import Sequence, Tuple

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir

from .fieldview_dataflow import FieldviewRegion
from .gtir_tasklet_codegen import GtirTaskletCodegen


class GtirToTasklet(eve.NodeVisitor):
    """Translates GTIR to Python code to be used as tasklet body.

    TODO: this class needs to be revisited in next commit.
    """

    _ctx: FieldviewRegion

    def __init__(self, ctx: FieldviewRegion):
        self._ctx = ctx

    def _visit_deref(self, node: itir.FunCall) -> str:
        # TODO: build memlet subset / shift pattern for each tasklet connector
        if not isinstance(node.args[0], itir.SymRef):
            raise NotImplementedError(
                f"Unexpected 'deref' argument with type '{type(node.args[0])}'."
            )
        return self.visit(node.args[0])

    def visit_Lambda(self, node: itir.Lambda) -> Tuple[str, Sequence[str], Sequence[str]]:
        params = [str(p.id) for p in node.params]
        results = []

        tlet_code_lines = []
        expr_list = GtirTaskletCodegen.apply(node.expr)
        for i, expr in enumerate(expr_list):
            outvar = f"__out_{i}"
            results.append(outvar)
            tlet_code_lines.append(outvar + " = " + expr)
        tlet_code = "\n".join(tlet_code_lines)

        return tlet_code, params, results
