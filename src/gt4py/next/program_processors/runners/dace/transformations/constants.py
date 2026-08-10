# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Constants needed by the optimizer."""

from typing import Final


GT_SIMPLIFY_DEFAULT_SKIP_SET: Final[set[str]] = {"ScalarToSymbolPromotion", "ConstantPropagation"}
"""Set of simplify passes `gt_simplify()` skips by default.

The following passes are included:
- `ScalarToSymbolPromotion`: The lowering has sometimes to turn a scalar into a
    symbol or vice versa and at a later point to invert this again. However, this
    pass has some problems with this pattern so for the time being it is disabled.
- `ConstantPropagation`: Same reasons as `ScalarToSymbolPromotion`.
"""


_GT_AUTO_OPT_INITIAL_STEP_SIMPLIFY_SKIP_LIST: Final[set[str]] = GT_SIMPLIFY_DEFAULT_SKIP_SET
"""Simplify stages disabled during the initial simplification.

For now it is an alias of the default skip list `GT_SIMPLIFY_DEFAULT_SKIP_SET`.
"""

_GT_AUTO_OPT_TOP_LEVEL_STAGE_SIMPLIFY_SKIP_LIST: Final[set[str]] = GT_SIMPLIFY_DEFAULT_SKIP_SET | {
    "InlineSDFGs",
    "InlineControlFlowRegions",
    "ControlFlowRaising",
    "OptionalArrayInference",
    "ConstantPropagation",
    "PruneEmptyConditionalBranches",
    "RemoveUnusedSymbols",
    "ReferenceToView",
    "ConsolidateEdges",
    "MapToCopy",
}
"""Simplify stages disabled during the optimization of dataflow of the top level Maps"."""


_GT_AUTO_OPT_INNER_DATAFLOW_STAGE_SIMPLIFY_SKIP_LIST: Final[set[str]] = (
    GT_SIMPLIFY_DEFAULT_SKIP_SET
    | {
        "FuseStates",
        "DeadDataflowElimination",
        "DeadStateElimination",
        "InlineSDFGs",
        "InlineControlFlowRegions",
        "ControlFlowRaising",
        "OptionalArrayInference",
        "ConstantPropagation",
        "PruneEmptyConditionalBranches",
        "RemoveUnusedSymbols",
        "ReferenceToView",
        "ContinueToCondition",
        "ConsolidateEdges",
        "GT4PyDeadDataflowElimination",
        "CopyChainRemover",
        "SingleStateGlobalDirectSelfCopyElimination",
        "SingleStateGlobalSelfCopyElimination",
        "MultiStateGlobalSelfCopyElimination",
        "MapToCopy",
    }
)
"""Simplify stages disabled during the optimization of dataflow inside the Maps."""


__all__ = ["GT_SIMPLIFY_DEFAULT_SKIP_SET"]
