# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from typing import Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_applied_lift
from gt4py.next.iterator.transforms.remap_symbols import RemapSymbolRefs, RenameSymbols
from gt4py.next.iterator.transforms.symbol_ref_utils import CountSymbolRefs


# TODO(tehrengruber): Reduce complexity of the function by removing the different options here
#  and introduce a generic predicate argument for the `eligible_params` instead.
def inline_lambda(  # noqa: C901  # see todo above
    node: ir.FunCall,
    opcount_preserving=False,
    force_inline_lift_args=False,
    force_inline_trivial_lift_args=False,
    force_inline_lambda_args=False,
    eligible_params: Optional[list[bool]] = None,
):
    assert isinstance(node.fun, ir.Lambda)
    eligible_params = eligible_params or [True] * len(node.fun.params)

    assert len(eligible_params) == len(node.fun.params) == len(node.args)

    if opcount_preserving:
        ref_counts = CountSymbolRefs.apply(node.fun.expr, [p.id for p in node.fun.params])

        for i, param in enumerate(node.fun.params):
            # TODO(tehrengruber): allow inlining more complicated zero-op expressions like
            #  ignore_shift(...)(it_sym)  # noqa: E800
            if ref_counts[param.id] > 1 and not isinstance(
                node.args[i], (ir.SymRef, ir.Literal, ir.OffsetLiteral)
            ):
                eligible_params[i] = False

    # inline lifts, i.e. `lift(λ(...) → ...)(...)`
    if force_inline_lift_args:
        for i, arg in enumerate(node.args):
            if is_applied_lift(arg):
                eligible_params[i] = True

    # inline trivial lifts, i.e. `lift(λ() → 1)()`
    if force_inline_trivial_lift_args:
        for i, arg in enumerate(node.args):
            if is_applied_lift(arg) and len(arg.args) == 0:
                eligible_params[i] = True

    # inline lambdas passed as arguments
    if force_inline_lambda_args:
        for i, arg in enumerate(node.args):
            if isinstance(arg, ir.Lambda):
                eligible_params[i] = True

    if node.fun.params and not any(eligible_params):
        return node

    refs = set().union(
        *(
            arg.pre_walk_values().if_isinstance(ir.SymRef).getattr("id").to_set()
            for arg, eligible in zip(node.args, eligible_params)
            if eligible
        )
    )
    syms = node.fun.expr.pre_walk_values().if_isinstance(ir.Sym).getattr("id").to_set()
    clashes = refs & syms
    expr = node.fun.expr
    if clashes:
        # TODO(tehrengruber): find a better way of generating new symbols
        #  in `name_map` that don't collide with each other. E.g. this
        #  must still work:
        # (lambda arg, arg_: (lambda arg_: ...)(arg))(a, b)  # noqa: E800
        name_map: dict[ir.SymRef, str] = {}

        def new_name(name):
            while name in refs or name in syms or name in name_map.values():
                name += "_"
            return name

        for sym in clashes:
            name_map[sym] = new_name(sym)

        expr = RenameSymbols().visit(expr, name_map=name_map)

    symbol_map = {
        param.id: arg
        for param, arg, eligible in zip(node.fun.params, node.args, eligible_params)
        if eligible
    }
    new_expr = RemapSymbolRefs().visit(expr, symbol_map=symbol_map)

    if all(eligible_params):
        new_expr.location = node.location
        return new_expr
    else:
        return ir.FunCall(
            fun=ir.Lambda(
                params=[
                    param
                    for param, eligible in zip(node.fun.params, eligible_params)
                    if not eligible
                ],
                expr=new_expr,
            ),
            args=[arg for arg, eligible in zip(node.args, eligible_params) if not eligible],
            location=node.location,
        )


@dataclasses.dataclass
class InlineLambdas(PreserveLocationVisitor, NodeTranslator):
    """Inline lambda calls by substituting every argument by its value."""

    PRESERVED_ANNEX_ATTRS = ("type",)

    opcount_preserving: bool

    force_inline_lambda_args: bool

    force_inline_lift_args: bool

    force_inline_trivial_lift_args: bool

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        opcount_preserving=False,
        force_inline_lambda_args=False,
        force_inline_lift_args=False,
        force_inline_trivial_lift_args=False,
    ):
        """
        Inline lambda calls by substituting every argument by its value.

        Examples:
            `(λ(x) → x)(y)` to `y`
            `(λ(x) → x)(y+y)` to `y+y`
            `(λ(x) → x+x)(y+y)` to `y+y+y+y` if not opcount_preserving
            `(λ(x) → x+x)(y+y)` stays as is if opcount_preserving

        Arguments:
            node: The function call node to inline into.
            opcount_preserving: Preserve the number of operations, i.e. only
                inline lambda call if the resulting call has the same number of
                operations.
            force_inline_lambda_args: Inline all arguments that are lambda calls, i.e.
                `(λ(p) → p(a, a))(λ(x, y) → x+y)`
            force_inline_lift_args: Inline all arguments that are applied lifts, i.e.
                `lift(λ(...) → ...)(...)`.
            force_inline_trivial_lift_args: Inline all arguments that are trivial
                applied lifts, e.g. `lift(λ() → 1)()`.

        """
        return cls(
            opcount_preserving=opcount_preserving,
            force_inline_lambda_args=force_inline_lambda_args,
            force_inline_lift_args=force_inline_lift_args,
            force_inline_trivial_lift_args=force_inline_trivial_lift_args,
        ).visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            return inline_lambda(
                node,
                opcount_preserving=self.opcount_preserving,
                force_inline_lambda_args=self.force_inline_lambda_args,
                force_inline_lift_args=self.force_inline_lift_args,
                force_inline_trivial_lift_args=self.force_inline_trivial_lift_args,
            )

        return node
