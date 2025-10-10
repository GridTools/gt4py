# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import KeysView

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils


def _filter_domain(domain: domain_utils.SymbolicDomain, dims: KeysView[common.Dimension]):
    return domain_utils.SymbolicDomain(
        grid_type=domain.grid_type,
        ranges={d: r for d, r in domain.ranges.items() if d in dims},
    )


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """ """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            cond_expr, tb, fb = node.args

            if tb == fb:
                # note: as long as we visited the args we have a copy here, so no need to copy again
                tb.annex.domain = node.annex.domain
                return tb

            cond = domain_utils.SymbolicDomain.from_expr(cond_expr)
            if cond.empty():
                return node.args[2]

            tb_domain, fb_domain = (
                _filter_domain(arg.annex.domain, cond.ranges.keys()) for arg in node.args[1:]
            )
            assert all(isinstance(d, domain_utils.SymbolicDomain) for d in (tb_domain, fb_domain))
            if tb_domain.empty():
                return node.args[2]
            if fb_domain.empty():
                return node.args[1]

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
