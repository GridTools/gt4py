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

from .auto_opt import gt_auto_optimize, gt_inline_nested_sdfg, gt_set_iteration_order, gt_simplify
from .gpu_utils import GPUSetBlockSize, gt_gpu_transformation, gt_set_gpu_blocksize
from .loop_blocking import LoopBlocking
from .map_orderer import MapIterationOrder
from .map_promoter import SerialMapPromoter
from .map_serial_fusion import SerialMapFusion


__all__ = [
    "GPUSetBlockSize",
    "LoopBlocking",
    "MapIterationOrder",
    "SerialMapFusion",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "gt_auto_optimize",
    "gt_gpu_transformation",
    "gt_inline_nested_sdfg",
    "gt_set_iteration_order",
    "gt_set_gpu_blocksize",
    "gt_simplify",
]
