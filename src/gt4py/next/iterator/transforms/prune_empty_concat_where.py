# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Self
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils


def _covers(branch: itir.Expr, domain: domain_utils.SymbolicDomain) -> bool:
    """Return whether `branch` can replace a `concat_where` accessed on `domain`."""
    branch_domain = branch.annex.domain
    # a branch domain is the selected region restricted to the dimensions of the branch, so
    #  fewer dimensions mean the branch would have to be broadcast back before replacing
    return (
        isinstance(branch_domain, domain_utils.SymbolicDomain)
        and branch_domain.ranges.keys() == domain.ranges.keys()
    )


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """
    Prune `concat_where` expression with one branch never being accessed.

    This pass requires domain inference to be executed before. In particular it relies on the
     condition being in the canonical form that domain inference already requires, i.e. bounded
     on exactly one side, as the complement of the condition is not defined otherwise.

    This pass the true and false branch values to be fields, not tuples of fields. Execute
     `gt4py.next.iterator.transforms.concat_where.expand_tuple_args` before.

    >>> from gt4py.next import common
    >>> from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
    >>> from gt4py.next.iterator.transforms import infer_domain
    >>> IDim = common.Dimension("IDim")
    >>> expr = im.concat_where(
    ...     im.domain(common.GridType.UNSTRUCTURED, {IDim: (10, itir.InfinityLiteral.POSITIVE)}),
    ...     "a",
    ...     "b",
    ... )
    >>> accessed = im.domain(common.GridType.UNSTRUCTURED, {IDim: (0, 10)})
    >>> expr, _ = infer_domain.infer_expr(
    ...     expr, domain_utils.SymbolicDomain.from_expr(accessed), offset_provider={}
    ... )
    >>> assert prune_empty_concat_where(expr) == im.ref("b")
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls: type[Self], node: PRG) -> PRG:
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            cond_expr, tb, fb = node.args

            if tb == fb:
                # note: as long as we visited the args we have a copy here, so no need to copy again
                tb.annex.domain = node.annex.domain
                return tb

            domain = node.annex.domain
            cond = domain_utils.SymbolicDomain.from_expr(cond_expr)
            if (
                isinstance(domain, domain_utils.SymbolicDomain)
                and cond.ranges.keys() <= domain.ranges.keys()
            ):
                for is_true_branch, other_branch in ((True, fb), (False, tb)):
                    if domain_utils.concat_where_branch_domain(
                        domain, cond, is_true_branch
                    ).empty() and _covers(other_branch, domain):
                        other_branch.annex.domain = domain
                        return other_branch

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
