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
from .dead_dataflow_elimination import gt_eliminate_dead_dataflow
from .gpu_utils import (
    GPUSetBlockSize,
    gt_gpu_transform_non_standard_memlet,
    gt_gpu_transformation,
    gt_set_gpu_blocksize,
)
from .local_double_buffering import gt_create_local_double_buffering
from .loop_blocking import LoopBlocking
from .map_fusion import MapFusion, MapFusionParallel, MapFusionSerial
from .map_orderer import MapIterationOrder, gt_set_iteration_order
from .map_promoter import SerialMapPromoter
from .move_dataflow_into_if_body import MoveDataflowIntoIfBody
from .redundant_array_removers import (
    CopyChainRemover,
    MultiStateGlobalSelfCopyElimination,
    SingleStateGlobalDirectSelfCopyElimination,
    SingleStateGlobalSelfCopyElimination,
    gt_multi_state_global_self_copy_elimination,
    gt_remove_copy_chain,
)
from .simplify import (
    GT_SIMPLIFY_DEFAULT_SKIP_SET,
    GT4PyMapBufferElimination,
    GT4PyMoveTaskletIntoMap,
    gt_inline_nested_sdfg,
    gt_reduce_distributed_buffering,
    gt_simplify,
    gt_substitute_compiletime_symbols,
)
from .split_access_nodes import SplitAccessNode, gt_split_access_nodes
from .state_fusion import GT4PyStateFusion
from .strides import (
    gt_change_transient_strides,
    gt_map_strides_to_dst_nested_sdfg,
    gt_map_strides_to_src_nested_sdfg,
    gt_propagate_strides_from_access_node,
    gt_propagate_strides_of,
)
from .utils import gt_find_constant_arguments, gt_make_transients_persistent


__all__ = [
    "GT_SIMPLIFY_DEFAULT_SKIP_SET",
    "CopyChainRemover",
    "GPUSetBlockSize",
    "GT4PyMapBufferElimination",
    "GT4PyMoveTaskletIntoMap",
    "GT4PyStateFusion",
    "LoopBlocking",
    "MapFusion",
    "MapFusionParallel",
    "MapFusionSerial",
    "MapIterationOrder",
    "MoveDataflowIntoIfBody",
    "MultiStateGlobalSelfCopyElimination",
    "SerialMapPromoter",
    "SerialMapPromoterGPU",
    "SingleStateGlobalDirectSelfCopyElimination",
    "SingleStateGlobalSelfCopyElimination",
    "SplitAccessNode",
    "gt_auto_optimize",
    "gt_change_transient_strides",
    "gt_create_local_double_buffering",
    "gt_eliminate_dead_dataflow",
    "gt_find_constant_arguments",
    "gt_gpu_transform_non_standard_memlet",
    "gt_gpu_transformation",
    "gt_inline_nested_sdfg",
    "gt_make_transients_persistent",
    "gt_map_strides_to_dst_nested_sdfg",
    "gt_map_strides_to_src_nested_sdfg",
    "gt_multi_state_global_self_copy_elimination",
    "gt_propagate_strides_from_access_node",
    "gt_propagate_strides_of",
    "gt_reduce_distributed_buffering",
    "gt_remove_copy_chain",
    "gt_set_gpu_blocksize",
    "gt_set_iteration_order",
    "gt_simplify",
    "gt_split_access_nodes",
    "gt_substitute_compiletime_symbols",
]
