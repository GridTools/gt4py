# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from .caches import (
    FillFlushToLocalKCaches,
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from .horizontal_execution_merging import HorizontalExecutionMerging, OnTheFlyMerging
from .inlining import MaskInlining
from .mask_stmt_merging import MaskStmtMerging
from .pruning import NoFieldAccessPruning, UnreachableStmtPruning
from .temporaries import LocalTemporariesToScalars, WriteBeforeReadTemporariesToScalars
from .vertical_loop_merging import AdjacentLoopMerging


__all__ = [
    "AdjacentLoopMerging",
    "FillFlushToLocalKCaches",
    "HorizontalExecutionMerging",
    "IJCacheDetection",
    "KCacheDetection",
    "LocalTemporariesToScalars",
    "MaskInlining",
    "MaskStmtMerging",
    "NoFieldAccessPruning",
    "OnTheFlyMerging",
    "PruneKCacheFills",
    "PruneKCacheFlushes",
    "UnreachableStmtPruning",
    "WriteBeforeReadTemporariesToScalars",
]
