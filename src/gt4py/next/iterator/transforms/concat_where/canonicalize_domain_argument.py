# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py.eve import PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.ir_utils.domain_utils import SymbolicDomain
from gt4py.next.iterator.transforms import fixed_point_transformation


def _range_complement(
    range_: domain_utils.SymbolicRange,
) -> tuple[domain_utils.SymbolicRange, domain_utils.SymbolicRange]:
    # `[a, b[` -> `[-inf, a[` ∪ `[b, inf[`  # noqa: RUF003
    assert not any(isinstance(b, itir.InfinityLiteral) for b in [range_.start, range_.stop])
    return (
        domain_utils.SymbolicRange(itir.InfinityLiteral.NEGATIVE, range_.start),
        domain_utils.SymbolicRange(range_.stop, itir.InfinityLiteral.POSITIVE),
    )


class _CanonicalizeDomainArgument(
    PreserveLocationVisitor, fixed_point_transformation.FixedPointTransformation
):
    """
    Transform `concat_where` expressions into their canonical form.

    The canonical form of a `concat_where(domain, tb, fb)` expression is an expression where
    `domain` is a simple domain expression, i.e. no union or intersection, which is unbounded in
    one and only one side, e.g. something like [-inf, 1] or [1, inf], but not [1, 2] or
    d1 | d2.  This choice of a canonical form ensures that the domain inference can infer a
    contiguous domain for `tb` and `fb` as the one-sided domain simply splits the contiguous
    domain on which the entire expression is accessed into two contiguous parts. Or more
    formally expressed:

    The domain `tb` is inferred by intersecting the domain of the entire `concat_where` expression
    with the domain argument. Intersection with a single bounded domain arg preserves the domain
    contiguity. The domain of `fb` is inferred by intersection of the entire domain with the
    complement of the domain argument. The complement of a single sided domain is another single
    sided domain, so then following the same argument as before the domain of `fb` is contiguous.
    To make this more concrete consider the `concat_where` expr is accessed on the domain [a, b]
    and its domain argument is [-inf, c] then the domain of `tb` is inferred to be [a, min(b, c)]
    and the domain of `fb` is [min(b, c), b].

    Description of the transformation:

    If the expression is not simple, but a union or intersection, e.g., [1, 2] | [3, 4], then this
    transformation first expands into a nested `concat_where` of simple domain expressions.
    In our example `concat_where([1, 2] | [3, 4], tb, fv)` is rewritten to
    `concat_where([1, 2], tb, concat_where([3, 4], tb, fb))`.
    If the expression is simple and bounded on both sides e.g. something like
    `concat_where([1, 2], tb, fb)` then the expression is rewritten into a union of simple
    domain expressions which are bounded on one side and unbounded in the other, namely
    `concat_where([-inf, 1] | [2, inf], fb, tb)`. Both transformations are applied until a fixed
    point is reached, ensuring first, a simple domain and second domain bounded on one side, in
    other words the desired canonical form.
    """

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def transform(self, node: itir.Node) -> Optional[itir.Node]:  # type: ignore[override] # ignore kwargs for simplicity
        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            # `concat_where(d1 & d2, a, b)` -> concat_where(d1, concat_where(d2, a, b), b)
            if cpm.is_call_to(cond_expr, "and_"):
                conds = cond_expr.args
                return im.let(("__cwcda_field_a", field_a), ("__cwcda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwcda_field_a", "__cwcda_field_b")
                            ),
                            "__cwcda_field_b",
                        )
                    )
                )
            # `concat_where(d1 | d2, a, b)` -> concat_where(d1, a, concat_where(d2, a, b))
            if cpm.is_call_to(cond_expr, "or_"):
                conds = cond_expr.args
                return im.let(("__cwcda_field_a", field_a), ("__cwcda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            "__cwcda_field_a",
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwcda_field_a", "__cwcda_field_b")
                            ),
                        )
                    )
                )

            # concat_where([1, 2[, a, b) -> concat_where([-inf, 1] | [2, inf[, b, a)
            if cpm.is_call_to(cond_expr, ("cartesian_domain", "unstructured_domain")):
                domain = SymbolicDomain.from_expr(cond_expr)
                if len(domain.ranges) == 1:
                    dim, range_ = next(iter(domain.ranges.items()))
                    if domain_utils.is_finite(range_):
                        complement = _range_complement(range_)
                        new_domains = [
                            im.domain(domain.grid_type, {dim: (cr.start, cr.stop)})
                            for cr in complement
                        ]
                        return self.fp_transform(
                            im.concat_where(im.call("or_")(*new_domains), field_b, field_a)
                        )
                else:
                    # TODO(tehrengruber): Implement. Note that this case can not be triggered by
                    #  the frontend yet since domains can only be created by expressions like
                    #  `IDim < 10`.
                    raise NotImplementedError()

        return None


canonicalize_domain_argument = _CanonicalizeDomainArgument.apply
