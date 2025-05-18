# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)


class ExpandLibraryFunctions(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)

        # `in_({i, j, k}, u⟨ Iₕ: [i0, i1[, Iₕ: [j0, j1[, Iₕ: [k0, k1[ ⟩`
        # -> `i0 <= i < i1 & j0 <= j < j1 & k0 <= k < k1`
        if cpm.is_call_to(node, "in_"):
            ret = []
            pos, domain = node.args
            for i, v in enumerate(
                domain_utils.SymbolicDomain.from_expr(node.args[1]).ranges.values()
            ):
                ret.append(
                    im.and_(
                        im.less_equal(v.start, im.tuple_get(i, pos)),
                        im.less(im.tuple_get(i, pos), v.stop),
                    )
                )  # TODO(tehrengruber): Avoid position expr duplication.
            return reduce(im.and_, ret)

        return node
