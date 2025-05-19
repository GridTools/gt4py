# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Optional

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_applied_lift
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.transforms.remap_symbols import RemapSymbolRefs
from gt4py.next.iterator.type_system import inference as itir_inference


# TODO(tehrengruber): Reduce complexity of the function by removing the different options here
#  and introduce a generic predicate argument for the `eligible_params` instead.
def inline_lambda(  # see todo above
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
        ref_counts = symbol_ref_utils.CountSymbolRefs.apply(
            node.fun.expr, [p.id for p in node.fun.params]
        )

        for i, param in enumerate(node.fun.params):
            # TODO(tehrengruber): allow inlining more complicated zero-op expressions like ignore_shift(...)(it_sym)
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

    symbol_map = {
        str(param.id): arg
        for param, arg, eligible in zip(node.fun.params, node.args, eligible_params)
        if eligible
    }

    new_fun_proto = im.lambda_(
        *(param for param, eligible in zip(node.fun.params, eligible_params) if not eligible)
    )(node.fun.expr)
    new_fun_proto = RemapSymbolRefs.apply(new_fun_proto, symbol_map=symbol_map)
    new_expr = im.call(new_fun_proto)(
        *(arg for arg, eligible in zip(node.args, eligible_params) if not eligible)
    )

    if all(eligible_params):
        new_expr = new_expr.fun.expr
        new_expr.location = node.location

    for attr in ("type", "recorded_shifts", "domain"):
        if hasattr(node.annex, attr):
            setattr(new_expr.annex, attr, getattr(node.annex, attr))
    itir_inference.copy_type(from_=node, to=new_expr, allow_untyped=True)
    return new_expr


@dataclasses.dataclass
class InlineLambdas(PreserveLocationVisitor, NodeTranslator):
    """
    Inline lambda calls by substituting every argument by its value.

    Note: This pass preserves, but doesn't use the `type` `recorded_shifts`, `domain` annex.
    """

    PRESERVED_ANNEX_ATTRS = ("type", "recorded_shifts", "domain")

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
