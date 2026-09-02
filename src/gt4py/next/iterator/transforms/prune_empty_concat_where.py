# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import Optional, TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Self
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


def _broadcast_to(expr: itir.Expr, target_dims: list[common.Dimension]) -> itir.Expr:
    """Broadcast `expr` to `target_dims` if it does not have these dimensions already."""
    assert isinstance(expr.type, (ts.FieldType, ts.ScalarType))
    if type_info.extract_dims(expr.type) != target_dims:
        return im.broadcast(expr, target_dims)
    return expr


def _concat_where_with_explicit_broadcast(
    node: itir.FunCall,
    *,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
) -> itir.FunCall:
    """
    Make the implicit broadcast in the branches of a `concat_where` explicit.

    The resulting expression is domain inferred, since the domain of a branch is restricted
    to the branch's own dimensions. E.g., in `concat_where(K < 0, a, b)` with
    `a: Field[[Vertex]]` accessed on
    `Vertex: [0, 10), K: [0, 10)` the domain of `a` is just `Vertex: [0, 10)` — the empty
    range `K: [0, 0)`, which decides that `a` is never selected, is dropped as `a` does not
    have the `K` dimension — while the domain of the corresponding
    `broadcast(a, (Vertex, K))` retains it. Domain inference instead of computing the
    domain of the broadcast is used for simplicity and to not duplicate the domain
    semantics of `concat_where`.
    """
    assert isinstance(node.type, (ts.FieldType, ts.ScalarType))
    target_dims = type_info.extract_dims(node.type)
    cond_expr, tb, fb = node.args
    new_node, _ = infer_domain.infer_expr(
        im.concat_where(
            cond_expr,
            _broadcast_to(tb, target_dims),
            _broadcast_to(fb, target_dims),
        ),
        node.annex.domain,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        revisit_already_inferred=False,
    )
    return new_node


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """
    Prune `concat_where` expressions with one branch never being selected.

    A branch is never selected exactly if its inferred domain, i.e. the intersection of the
    domain of the `concat_where` with the condition (respectively its complement for the false
    branch), is empty.

    This pass requires domain and type inference to be executed before.

    This pass only applies if the true and false branch values are fields, not tuples of
    fields; `concat_where` expressions on tuples are left untouched. Execute
    `gt4py.next.iterator.transforms.concat_where.expand_tuple_args` before to prune them.

    >>> from gt4py.next import common
    >>> IDim = common.dimension("IDim")
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
    >>> assert prune_empty_concat_where(expr, offset_provider={}) == im.ref("b")
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    offset_provider: common.OffsetProvider | common.OffsetProviderType
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None

    @classmethod
    def apply(
        cls: type[Self],
        node: PRG,
        *,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
        symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    ) -> PRG:
        return cls(
            offset_provider=offset_provider, symbolic_domain_sizes=symbolic_domain_sizes
        ).visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            type_inference.reinfer(node)
            if not isinstance(node.type, (ts.FieldType, ts.ScalarType)):
                return node

            _, tb, fb = _concat_where_with_explicit_broadcast(
                node,
                offset_provider=self.offset_provider,
                symbolic_domain_sizes=self.symbolic_domain_sizes,
            ).args

            if tb == fb:
                # The branch is selected on the entire domain of the `concat_where`, but its
                #  domains were inferred from the region where each branch instance is selected.
                #  Reinfer on the full domain, e.g. in `concat_where(2 <= K, a, broadcast(a,
                #  (Vertex, K)))` the `a` inside the surviving `broadcast` would otherwise keep
                #  the domain `K: [0, 2)` of the true branch.
                # note: as long as we visited the args we have a copy here, so no need to copy again
                new_node, _ = infer_domain.infer_expr(
                    tb,
                    node.annex.domain,
                    keep_existing_domains=False,
                    revisit_already_inferred=True,
                    offset_provider=self.offset_provider,
                    symbolic_domain_sizes=self.symbolic_domain_sizes,
                )
                return new_node

            if not isinstance(node.annex.domain, domain_utils.SymbolicDomain):
                return node

            if tb.annex.domain.empty():
                return fb
            if fb.annex.domain.empty():
                return tb

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
