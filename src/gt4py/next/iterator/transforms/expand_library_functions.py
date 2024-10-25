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
    @classmethod
    def apply(cls, node: ir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        if cpm.is_call_to(node, "in"):
            ret = []
            pos, domain = node.args
            for i, (k, v) in enumerate(
                domain_utils.SymbolicDomain.from_expr(node.args[1]).ranges.items()
            ):
                ret.append(
                    im.and_(
                        im.less_equal(v.start, im.tuple_get(i, pos)),
                        im.less(im.tuple_get(i, pos), v.stop),
                    )
                )  # TODO: avoid pos duplication
            return reduce(im.and_, ret)
        return self.generic_visit(node)
