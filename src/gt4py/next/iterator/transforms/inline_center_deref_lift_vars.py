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
from typing import ClassVar, Optional

import gt4py.next.iterator.ir_utils.common_pattern_matcher as common_pattern_matcher
from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda
from gt4py.next.iterator.transforms.inline_lifts import InlineLifts
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts, copy_recorded_shifts


def is_center_derefed_only(node: itir.Node) -> bool:
    return hasattr(node.annex, "recorded_shifts") and node.annex.recorded_shifts in [set(), {()}]


@dataclasses.dataclass
class InlineCenterDerefLiftVars(eve.NodeTranslator):
    """
    Inline all variables which are derefed in the center only (i.e. unshifted).

    Consider the following example where `var` is never shifted:

    `let(var, (↑stencil)(it))(·var + ·var)`

    Directly inlining `var` would increase the size of the tree and duplicate the calculation.
    Instead, this pass computes the value at the current location once and replaces all previous
    references to `var` by an applied lift which captures this value.

    `let(_icdlv_1, stencil(it))(·(↑(λ() → _icdlv_1) + ·(↑(λ() → _icdlv_1))`

    The lift inliner can then later easily transform this into a nice expression:

    `let(_icdlv_1, stencil(it))(_icdlv_1 + _icdlv_1)`

    Note: This pass uses and preserves the `recorded_shifts` annex.
    """

    PRESERVED_ANNEX_ATTRS: ClassVar[tuple[str, ...]] = ("recorded_shifts",)

    uids: eve_utils.UIDGenerator

    @classmethod
    def apply(cls, node: itir.FencilDefinition, uids: Optional[eve_utils.UIDGenerator] = None):
        if not uids:
            uids = eve_utils.UIDGenerator()
        return cls(uids=uids).visit(node)

    def visit_StencilClosure(self, node: itir.StencilClosure, **kwargs):
        # TODO(tehrengruber): move the analysis out of this pass and just make it a requirement
        #  such that we don't need to run in multiple times if multiple passes use it.
        TraceShifts.apply(node, save_to_annex=True)
        return self.generic_visit(node, **kwargs)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node)
        if common_pattern_matcher.is_let(node):
            assert isinstance(node.fun, itir.Lambda)  # to make mypy happy
            eligible_params = [False] * len(node.fun.params)
            new_args = []
            bound_scalars: dict[str, itir.Expr] = {}

            for i, (param, arg) in enumerate(zip(node.fun.params, node.args)):
                if common_pattern_matcher.is_applied_lift(arg) and is_center_derefed_only(param):
                    eligible_params[i] = True
                    bound_arg_name = self.uids.sequential_id(prefix="_icdlv")
                    capture_lift = im.promote_to_const_iterator(bound_arg_name)
                    copy_recorded_shifts(from_=param, to=capture_lift)
                    new_args.append(capture_lift)
                    # since we deref an applied lift here we can (but don't need to) immediately
                    # inline
                    bound_scalars[bound_arg_name] = InlineLifts(
                        flags=InlineLifts.Flag.INLINE_TRIVIAL_DEREF_LIFT
                    ).visit(im.deref(arg), recurse=False)
                else:
                    new_args.append(arg)

            if any(eligible_params):
                new_node = inline_lambda(
                    im.call(node.fun)(*new_args), eligible_params=eligible_params
                )
                # TODO(tehrengruber): propagate let outwards
                return im.let(*bound_scalars.items())(new_node)  # type: ignore[arg-type] # mypy not smart enough

        return node
