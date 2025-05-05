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
    ir: itir.Program,
    *,
    collapse_tuple_uids: Optional[eve_utils.UIDGenerator] = None,
    offset_provider_type: common.OffsetProviderType,
) -> itir.Program:
    # remove dead-code, e.g. `let var=True in if_(var, val1, val2) end`
    # ensure all constant let init forms are inlined
    # `let var=True in if_(var, val1, val2) end` -> `if_(True, val1, val2)`
    ir = InlineLambdas.apply(ir, opcount_preserving=True)

    # remove the unreachable if branches
    # `if_(True, val1, val2)` -> `val1`
    ir = ConstantFolding.apply(ir, enabled_transformations=ConstantFolding.Transformation.FOLD_IF)  # type: ignore[assignment]  # always an itir.Program

    # note: `force_inline_lambda_args` increases the size of the tree and may not be required for
    # dead-code-elimination, but is needed since the domain inference cannot handle "user"
    # functions, e.g. `let f = λ(...) → ... in f(...)`
    ir = InlineLambdas.apply(ir, opcount_preserving=True, force_inline_lambda_args=True)

    # get rid of tuple elements that are never accessed
    # `{a, b}[0]` -> `a`
    ir = CollapseTuple.apply(
        ir,
        enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
        uids=collapse_tuple_uids,
        offset_provider_type=offset_provider_type,
    )  # type: ignore[assignment]  # always an itir.Program

    return ir
