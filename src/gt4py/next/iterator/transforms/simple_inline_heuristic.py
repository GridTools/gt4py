# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_lifts import _is_lift


def is_eligible_for_inlining(node: ir.FunCall, is_scan_pass_context: bool) -> bool:
    """
    Predicate for the InlineLifts transformation.

    Takes a lifted stencil call of the form `↑(f)(args...)` and returns whether the expression
    is eligible for inlining. For example `·↑(f)(args...)` would be transformed into
    `f(args...)` if true.

    The ``is_scan_pass_context`` argument indicates if the given node is within a scan (unnested),
    e.g. `↑(f)(args...)` should not be inlined if it appears in a scan like this:
    `↑(scan(λ(acc, args...) → acc + ·↑(f)(args...)))(...)`

    Follows the simple rules:
    - Do not inline scans (as there is no efficient way to inline them, also required by some
      backends, e.g. gtfn)
    - Do not inline the first lifted function call within a scan (otherwise, all stencils would get
      inlined into the scans, leading to reduced parallelism/scan-only computation)
    """
    assert _is_lift(node)

    assert isinstance(node.fun, ir.FunCall)  # for mypy
    (stencil,) = node.fun.args
    # Don't inline scans, i.e. exclude `↑(scan(...))(...)`
    if isinstance(stencil, ir.FunCall) and stencil.fun == ir.SymRef(id="scan"):
        return False

    # Don't inline the first lifted function call within a scan, e.g. if the node given here
    # is `↑(f)(args...)` and appears in a scan pass `scan(λ(acc, args...) → acc + ·↑(f)(args...))`
    # it should not be inlined.
    return not is_scan_pass_context
