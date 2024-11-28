# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.symbol_ref_utils import CountSymbolRefs


class MergeLet(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Merge let-like statements.

    For example transforms::

        (λ(a) → (λ(b) → a+b)(arg1))(arg2)

    into::

        (λ(a, b) → a+b)(arg1, arg2)

    This can significantly reduce the depth of the tree and its readability.
    """

    def visit_FunCall(self, node: itir.FunCall):
        node = self.generic_visit(node)
        if (
            isinstance(node.fun, itir.Lambda)
            and isinstance(node.fun.expr, itir.FunCall)
            and isinstance(node.fun.expr.fun, itir.Lambda)
        ):
            outer_lambda = node.fun
            outer_lambda_args = node.args
            inner_lambda = node.fun.expr.fun
            inner_lambda_args = node.fun.expr.args
            # skip if we have a collision
            if set(outer_lambda.params) & set(inner_lambda.params):
                return node
            # check if the arguments to the outer lambda call use a symbol of the inner lambda
            # e.g. (λ(a) → (λ(b) → b)(a))(b)
            ref_counts_outer = CountSymbolRefs.apply(
                outer_lambda_args, [param.id for param in inner_lambda.params]
            )
            if any(ref_count != 0 for ref_count in ref_counts_outer.values()):
                return node
            # check if the argument to the inner lambda call depend on an argument to the outer lambda
            ref_counts = CountSymbolRefs.apply(
                inner_lambda_args, [param.id for param in outer_lambda.params]
            )
            if any(ref_count != 0 for ref_count in ref_counts.values()):
                return node
            return itir.FunCall(
                fun=itir.Lambda(
                    params=outer_lambda.params + inner_lambda.params, expr=inner_lambda.expr
                ),
                args=outer_lambda_args + inner_lambda_args,
            )
        return node
