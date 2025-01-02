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
from .gpu_utils import (
    GPUSetBlockSize,
    gt_gpu_transform_non_standard_memlet,
    gt_gpu_transformation,
    gt_set_gpu_blocksize,
)
from .local_double_buffering import gt_create_local_double_buffering
from .loop_blocking import LoopBlocking
from .map_fusion_parallel import MapFusionParallel
from .map_fusion_serial import MapFusionSerial
from .map_orderer import MapIterationOrder, gt_set_iteration_order
from .map_promoter import SerialMapPromoter
from .simplify import (
    GT_SIMPLIFY_DEFAULT_SKIP_SET,
    GT4PyGlobalSelfCopyElimination,
    GT4PyMapBufferElimination,
    GT4PyMoveTaskletIntoMap,
    gt_inline_nested_sdfg,
    gt_reduce_distributed_buffering,
    gt_simplify,
    gt_substitute_compiletime_symbols,
)
from .strides import (
    gt_change_transient_strides,
    gt_map_strides_to_dst_nested_sdfg,
    gt_map_strides_to_src_nested_sdfg,
    gt_propagate_strides_from_access_node,
    gt_propagate_strides_of,
)
from .util import gt_find_constant_arguments, gt_make_transients_persistent


__all__ = [
    "GT_SIMPLIFY_DEFAULT_SKIP_SET",
    "GPUSetBlockSize",
    "GT4PyGlobalSelfCopyElimination",
    "GT4PyMapBufferElimination",
    "GT4PyMoveTaskletIntoMap",
    "LoopBlocking",
    "MapFusionParallel",
    "MapFusionSerial",
    "MapIterationOrder",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "gt_auto_optimize",
    "gt_change_transient_strides",
    "gt_create_local_double_buffering",
    "gt_find_constant_arguments",
    "gt_gpu_transform_non_standard_memlet",
    "gt_gpu_transformation",
    "gt_inline_nested_sdfg",
    "gt_make_transients_persistent",
    "gt_map_strides_to_dst_nested_sdfg",
    "gt_map_strides_to_src_nested_sdfg",
    "gt_propagate_strides_from_access_node",
    "gt_propagate_strides_of",
    "gt_reduce_distributed_buffering",
    "gt_set_gpu_blocksize",
    "gt_set_iteration_order",
    "gt_simplify",
    "gt_substitute_compiletime_symbols",
]
