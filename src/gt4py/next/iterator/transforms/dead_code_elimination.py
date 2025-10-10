# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


def dead_code_elimination(
    program: itir.Program,
    *,
    collapse_tuple_uids: Optional[eve_utils.UIDGenerator] = None,
    offset_provider_type: common.OffsetProviderType,
) -> itir.Program:
    """
    Perform dead code elimination on a program by simplifying or removing
    unused or unreachable code constructs.

    This transformation is a composition of InlineLambdas, ConstantFolding, and CollapseTuple.

    Example:
    ```
    let val = {inp1, inp2}[0]
      if_ True then
        val
      else
        inp3
    end
    ```
    is transformed into
    ```
    inp1
    ```

    Note: `concat_where.expand_tuple_args` is required to be executed before, in order to eliminate
    dead-code in cases like:
    ```
    let tmp = concat_where(cond, {a, b, c}, {d, e, f})
      non_trivial_expr(tmp[0], tmp[2])
    end
    ```
    That `tmp` is referenced twice in this example is important as otherwise function inlining
    and `tuple_get` propagation would also remove the tuple. Additionally, `non_trivial_expr`,
    must be a non-trivial expression as otherwise cps tuple inlining would also resolve this case.
    """
    # ensure all constant let bindings are inlined
    # `let var=True in if_(var, val1, val2) end` -> `if_(True, val1, val2)`
    # TODO(tehrengruber): we only want to inline literals here
    program = InlineLambdas.apply(program, opcount_preserving=True)

    # remove the unreachable if branches
    # e.g. `if_(True, val1, val2)` -> `val1`
    program = ConstantFolding.apply(program)  # type: ignore[assignment]  # always an itir.Program

    # inline again since after constant folding some expressions might not be referenced anymore,
    # e.g. `let field = as_fieldop(...) in val1 end` -> `val1`.
    # TODO(tehrengruber): If we first re-arrange the tree such that let bindings are placed as
    #  close as possible to references to them we don't need to inline again here.
    # note: `force_inline_lambda_args` increases the size of the tree and may not be required for
    # dead-code-elimination, but is needed later since the domain inference cannot handle "user"
    # functions, e.g. `let f = λ(...) → ... in f(...)`
    program = InlineLambdas.apply(program, opcount_preserving=True, force_inline_lambda_args=True)

    # get rid of tuple elements that are never accessed
    # `{a, b}[0]` -> `a`
    program = CollapseTuple.apply(
        program,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        uids=collapse_tuple_uids,
        offset_provider_type=offset_provider_type,
    )  # type: ignore[assignment]  # always an itir.Program

    return program
