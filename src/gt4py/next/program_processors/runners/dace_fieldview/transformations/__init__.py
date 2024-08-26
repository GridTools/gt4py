# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformation and optimization pipeline for the DaCe backend in GT4Py.

Please also see [this HackMD document](https://hackmd.io/@gridtools/rklwk4OIR#Requirements-on-SDFG)
that explains the general structure and requirements on the SDFG.
"""

from .auto_opt import gt_auto_optimize, gt_set_iteration_order, gt_simplify
from .gpu_utils import (
    GPUSetBlockSize,
    SerialMapPromoterGPU,
    gt_gpu_transformation,
    gt_set_gpu_blocksize,
)
from .k_blocking import KBlocking
from .map_orderer import MapIterationOrder
from .map_promoter import SerialMapPromoter
from .map_serial_fusion import SerialMapFusion


__all__ = [
    "GPUSetBlockSize",
    "KBlocking",
    "MapIterationOrder",
    "SerialMapFusion",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "gt_auto_optimize",
    "gt_gpu_transformation",
    "gt_set_iteration_order",
    "gt_set_gpu_blocksize",
    "gt_simplify",
]
