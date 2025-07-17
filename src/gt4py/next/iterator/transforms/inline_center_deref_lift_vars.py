# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import ClassVar, Optional, TypeVar

import gt4py.next.iterator.ir_utils.common_pattern_matcher as cpm
from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import trace_shifts
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts


def is_center_derefed_only(node: itir.Node) -> bool:
    return hasattr(node.annex, "recorded_shifts") and node.annex.recorded_shifts in [set(), {()}]


T = TypeVar("T", bound=itir.Node)


@dataclasses.dataclass
class InlineCenterDerefLiftVars(eve.NodeTranslator):
    """
    Inline all variables which are derefed in the center only (i.e. unshifted).

    Consider the following example where `var` is never shifted:

    `let(var, (↑stencil)(it))(·var + ·var)`

    Directly inlining `var` would increase the size of the tree and duplicate the calculation.
    Instead this pass, first takes the iterator `(↑stencil)(it)` and transforms it into a
    0-ary function that evaluates to the value at the current location.

    `λ() → ·(↑stencil)(it)`

    Then all previous occurences of `var` are replaced by this function.

    `let(_icdlv_1, λ() → ·(↑stencil)(it))(·(↑(λ() → _icdlv_1()) + ·(↑(λ() → _icdlv_1()))`

    The lift inliner can then later easily transform this into a nice expression:

     `let(_icdlv_1, λ() → stencil(it))(_icdlv_1() + _icdlv_1())`

    Finally, recomputation is avoided by using the common subexpression elimination and lamba
    inlining (can be configured opcount preserving). Both is up to the caller to do later.

    `λ(_cs_1) → _cs_1 + _cs_1)(stencil(it))`

    Note: This pass uses and preserves the `domain` and `recorded_shifts` annex.
    """

    PRESERVED_ANNEX_ATTRS: ClassVar[tuple[str, ...]] = ("domain", "recorded_shifts")

    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(
        cls, node: T, *, is_stencil=False, uids: Optional[eve_utils.UIDGenerator] = None
    ) -> T:
        if not uids:
            uids = eve_utils.UIDGenerator()
        if is_stencil:
            assert isinstance(node, itir.Expr)
            trace_shifts.trace_stencil(node, save_to_annex=True)
        return cls(uids=uids).visit(node)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        # TODO(tehrengruber): move the analysis out of this pass and just make it a requirement
        #  such that we don't need to run in multiple times if multiple passes use it.
        if cpm.is_call_to(node.fun, "as_fieldop"):
            assert len(node.fun.args) in [1, 2]
            trace_shifts.trace_stencil(
                node.fun.args[0], num_args=len(node.args), save_to_annex=True
            )

        node = self.generic_visit(node)

        if cpm.is_let(node):
            assert isinstance(node.fun, itir.Lambda)  # to make mypy happy
            eligible_params = [False] * len(node.fun.params)
            new_args = []
            # values are 0-ary lambda functions that evaluate to the derefed argument. We don't put
            # the values themselves here as they might be inside of an if to protected from an oob
            # access
            evaluators: dict[str, itir.Expr] = {}

            for i, (param, arg) in enumerate(zip(node.fun.params, node.args)):
                if cpm.is_applied_lift(arg) and is_center_derefed_only(param):
                    eligible_params[i] = True
                    bound_arg_evaluator = self.uids.sequential_id(prefix="_icdlv")
                    capture_lift = im.promote_to_const_iterator(im.call(bound_arg_evaluator)())
                    trace_shifts.copy_recorded_shifts(from_=param, to=capture_lift)
                    new_args.append(capture_lift)
                    # since we deref an applied lift here we can (but don't need to) immediately
                    # inline
                    evaluators[bound_arg_evaluator] = im.lambda_()(
                        InlineLifts(flags=InlineLifts.Flag.INLINE_DEREF_LIFT).visit(
                            im.deref(arg), recurse=False
                        )
                    )
                else:
                    new_args.append(arg)

            if any(eligible_params):
                new_node = inline_lambda(
                    im.call(node.fun)(*new_args), eligible_params=eligible_params
                )
                # TODO(tehrengruber): propagate let outwards
                return im.let(*evaluators.items())(new_node)

        return node
