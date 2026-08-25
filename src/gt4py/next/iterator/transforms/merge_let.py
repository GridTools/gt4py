# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import ClassVar

import gt4py.eve as eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import misc as ir_misc
from gt4py.next.iterator.transforms.remap_symbols import RenameSymbols
from gt4py.next.iterator.transforms.symbol_ref_utils import CountSymbolRefs


class MergeLet(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Merge let-like statements.

    For example transforms::

        (λ(a) → (λ(b) → a+b)(arg1))(arg2)

    into::

        (λ(a, b) → a+b)(arg1, arg2)

    This can significantly reduce the depth of the tree and its readability.

    Parameters of the inner lambda that clash with a name bound or referenced by the outer
    call are renamed apart, such that the result only depends on the program up to renaming
    of its bound symbols. The merge is skipped only when an argument of the inner call
    references a parameter of the outer lambda, as hoisting it would move that argument out
    of the binder it refers to.
    """

    PRESERVED_ANNEX_ATTRS: ClassVar[tuple[str, ...]] = ("domain",)

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
            # check if the argument to the inner lambda call depend on an argument to the outer
            # lambda; hoisting such an argument would move it out of the binder it refers to
            ref_counts = CountSymbolRefs.apply(
                inner_lambda_args, [param.id for param in outer_lambda.params]
            )
            if any(ref_count != 0 for ref_count in ref_counts.values()):
                return node
            # Rename inner parameters that clash with a name bound by the outer lambda or
            # spelled by one of its arguments, instead of skipping the merge. Merging only
            # when the binders happen to be spelled apart would make the result depend on
            # the choice of names rather than on the alpha-equivalence class of the program.
            reserved = {param.id for param in outer_lambda.params} | {
                ref.id for ref in eve.walk_values(outer_lambda_args).if_isinstance(itir.SymRef)
            }
            clashes = {param.id for param in inner_lambda.params} & reserved
            if clashes:
                taken = reserved | (
                    eve.walk_values(inner_lambda)
                    .if_isinstance(itir.Sym, itir.SymRef)
                    .getattr("id")
                    .to_set()
                )
                name_map: dict[str, str] = {}
                for sym in sorted(clashes):
                    name_map[sym] = ir_misc.unique_symbol(sym, taken | {*name_map.values()})
                inner_lambda = RenameSymbols().visit(inner_lambda, name_map=name_map)
            return itir.FunCall(
                fun=itir.Lambda(
                    params=outer_lambda.params + inner_lambda.params, expr=inner_lambda.expr
                ),
                args=outer_lambda_args + inner_lambda_args,
            )
        return node
