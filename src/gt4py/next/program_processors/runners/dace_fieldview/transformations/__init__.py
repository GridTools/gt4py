# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformation and optimization pipeline for the DaCe backend in GT4Py.

Please also see [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md)
that explains the general structure and requirements on the SDFGs.
"""

from .auto_optimize import gt_auto_optimize
from .gpu_utils import GPUSetBlockSize, gt_gpu_transformation, gt_set_gpu_blocksize
from .loop_blocking import LoopBlocking
from .map_fusion_parallel import MapFusionParallel
from .map_fusion_serial import MapFusionSerial
from .map_orderer import MapIterationOrder
from .map_promoter import SerialMapPromoter
from .simplify import (
    GT_SIMPLIFY_DEFAULT_SKIP_SET,
    GT4PyGlobalSelfCopyElimination,
    GT4PyMoveTaskletIntoMap,
    GT4PyRedundantArrayElimination,
    gt_inline_nested_sdfg,
    gt_reduce_distributed_buffering,
    gt_set_iteration_order,
    gt_simplify,
    gt_substitute_compiletime_symbols,
)
from .util import gt_find_constant_arguments, gt_make_transients_persistent


__all__ = [
    "GT_SIMPLIFY_DEFAULT_SKIP_SET",
    "GPUSetBlockSize",
    "GT4PyGlobalSelfCopyElimination",
    "GT4PyRedundantArrayElimination",
    "GT4PyMoveTaskletIntoMap",
    "LoopBlocking",
    "MapIterationOrder",
    "MapFusionParallel",
    "MapFusionSerial",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "gt_auto_optimize",
    "gt_gpu_transformation",
    "gt_inline_nested_sdfg",
    "gt_set_iteration_order",
    "gt_set_gpu_blocksize",
    "gt_simplify",
    "gt_make_transients_persistent",
    "gt_reduce_distributed_buffering",
    "gt_find_constant_arguments",
    "gt_substitute_compiletime_symbols",
]
