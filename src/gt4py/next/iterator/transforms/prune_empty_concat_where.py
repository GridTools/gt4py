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
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """
    Prune `concat_where` expressions with one branch never being selected.

    Whether a branch is selected is decided from the domain of the `concat_where` itself:
    the intersection with the condition (respectively its complement for the false branch)
    is empty exactly if the branch is never selected.

    Since a `concat_where` implicitly broadcasts a branch to the dimensions it does not have
    itself, a surviving branch with fewer dimensions than the `concat_where` is wrapped in a
    `broadcast` call. The domain of the resulting expression is populated from the pruned
    `concat_where` (`RemoveBroadcast` relies on it to lower the `broadcast` again).

    This pass requires domain and type inference to be executed before. In particular it relies
    on the condition being in the canonical form that domain inference already requires, i.e.
    bounded on exactly one side, as the complement of the condition is not defined otherwise.

    This pass requires the true and false branch values to be fields, not tuples of fields.
    Execute `gt4py.next.iterator.transforms.concat_where.expand_tuple_args` before.

    >>> from gt4py.next import common
    >>> from gt4py.next.iterator.transforms import infer_domain
    >>> IDim = common.Dimension("IDim")
    >>> field_t = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    >>> expr = im.concat_where(
    ...     im.domain(common.GridType.CARTESIAN, {IDim: (10, itir.InfinityLiteral.POSITIVE)}),
    ...     im.ref("a", field_t),
    ...     im.ref("b", field_t),
    ... )
    >>> expr, _ = infer_domain.infer_expr(
    ...     expr,
    ...     domain_utils.SymbolicDomain.from_expr(
    ...         im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    ...     ),
    ...     offset_provider={},
    ... )
    >>> assert prune_empty_concat_where(expr) == im.ref("b")
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls: type[Self], node: PRG) -> PRG:
        return cls().visit(node)

    def _prune_to(self, node: itir.FunCall, branch: itir.Expr) -> itir.Expr:
        """Replace `node` by `branch`, broadcasting `branch` if it has fewer dimensions."""
        assert isinstance(node.type, ts.FieldType)  # `concat_where` has at least the concat dim
        assert isinstance(branch.type, (ts.FieldType, ts.ScalarType))
        if type_info.extract_dims(branch.type) != node.type.dims:
            branch = im.call("broadcast")(
                branch,
                im.make_tuple(*(im.axis_literal(dim) for dim in node.type.dims)),
            )
        branch.annex.domain = node.annex.domain
        return branch

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            # no `offset_provider_type` needed as we only infer field-view level expressions
            type_inference.reinfer(node)
            if not isinstance(node.type, (ts.FieldType, ts.ScalarType)):
                # TODO(tehrengruber): Implement support for tuples.
                return node

            cond_expr, tb, fb = node.args

            if tb == fb:
                # note: as long as we visited the args we have a copy here, so no need to copy again
                return self._prune_to(node, tb)

            accessed_domain = node.annex.domain
            cond = domain_utils.SymbolicDomain.from_expr(cond_expr)
            if isinstance(accessed_domain, domain_utils.SymbolicDomain):
                assert cond.ranges.keys() <= accessed_domain.ranges.keys()
                for branch, other_branch, selected_domain in (
                    (tb, fb, cond),
                    (fb, tb, domain_utils.domain_complement(cond)),
                ):
                    branch_accessed_domain = domain_utils.domain_intersection(
                        domain_utils.promote_domain(
                            branch.annex.domain, accessed_domain.ranges.keys()
                        ),
                        domain_utils.promote_domain(selected_domain, accessed_domain.ranges.keys()),
                        accessed_domain,
                    )
                    if branch_accessed_domain.empty():
                        return self._prune_to(node, other_branch)

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
