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

"""Transformation and optimization pipeline for the DaCe backend in GT4Py.

Please also see [this HackMD document](https://hackmd.io/@gridtools/rklwk4OIR#Requirements-on-SDFG)
that explains the general structure and requirements on the SDFG.
"""

from .auto_opt import dace_auto_optimize, gt_auto_optimize, gt_set_iteration_order, gt_simplify
from .gpu_utils import (
    GPUSetBlockSize,
    SerialMapPromoterGPU,
    gt_gpu_transformation,
    gt_set_gpu_blocksize,
)
from .k_blocking import KBlocking
from .map_fusion_serial import SerialMapFusion
from .map_orderer import MapIterationOrder
from .map_promoter import SerialMapPromoter


__all__ = [
    "GPUSetBlockSize",
    "KBlocking",
    "MapIterationOrder",
    "SerialMapFusion",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "dace_auto_optimize",
    "gt_auto_optimize",
    "gt_gpu_transformation",
    "gt_set_iteration_order",
    "gt_set_gpu_blocksize",
    "gt_simplify",
]
