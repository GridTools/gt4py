# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


if TYPE_CHECKING:
    # Type-only: a top-level import would cycle with ``compilation``.
    from gt4py.next.program_processors.runners.dace.workflow.compilation import CompiledDaceProgram


def convert_args(
    fun: CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> stages.ExecutableProgram:
    # Retieve metrics level from GT4Py environment variable.
    collect_time = metrics.is_level_enabled(metrics.PERFORMANCE)
    collect_time_arg = np.array(
        [1], dtype=gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME_DTYPE.as_numpy_dtype()
    )
    argument_preprocessing_function = fun.argument_preprocessing_function

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        processed_args = argument_preprocessing_function(
            args,
            offset_provider,
            metrics.get_current_level(),
            collect_time_arg,
        )
        fun.sdfg_program.user_bind_call(*processed_args)

        if collect_time:
            metrics.add_sample_to_current_source(metrics.COMPUTE_METRIC, collect_time_arg[0].item())

    return decorated_program
