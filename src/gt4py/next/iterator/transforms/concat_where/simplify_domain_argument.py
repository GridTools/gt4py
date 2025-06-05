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


class _SimplifyDomainArgument(
    PreserveLocationVisitor, fixed_point_transformation.FixedPointTransformation
):
    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def transform(self, node: itir.Node) -> Optional[itir.Node]:  # type: ignore[override] # ignore kwargs for simplicity
        if cpm.is_call_to(node, "concat_where"):
            cond_expr, field_a, field_b = node.args
            if cpm.is_call_to(cond_expr, "and_"):
                conds = cond_expr.args
                return im.let(("__cwsda_field_a", field_a), ("__cwsda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwsda_field_a", "__cwsda_field_b")
                            ),
                            "__cwsda_field_b",
                        )
                    )
                )
            if cpm.is_call_to(cond_expr, "or_"):
                conds = cond_expr.args
                return im.let(("__cwsda_field_a", field_a), ("__cwsda_field_b", field_b))(
                    self.fp_transform(
                        im.concat_where(
                            conds[0],
                            "__cwsda_field_a",
                            self.fp_transform(
                                im.concat_where(conds[1], "__cwsda_field_a", "__cwsda_field_b")
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


simplify_domain_argument = _SimplifyDomainArgument.apply
