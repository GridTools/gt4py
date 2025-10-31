# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import Any, Sequence

import dace
import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config, metrics, utils as gtx_utils
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace import sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    compilation as gtx_wfdcompilation,
)


def convert_args(
    fun: gtx_wfdcompilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> stages.CompiledProgram:
    # Retieve metrics level from GT4Py environment variable.
    collect_time = config.COLLECT_METRICS_LEVEL >= metrics.PERFORMANCE
    collect_time_arg = np.array([1], dtype=np.float64)
    # We use the callback function provided by the compiled program to update the SDFG arglist.
    update_sdfg_call_args = functools.partial(
        fun.update_sdfg_ctype_arglist, device, fun.sdfg_argtypes
    )

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        try:
            # Initialization of `_lastargs` was done by the `CompiledSDFG` object,
            #  so we just update it with the current call arguments.
            update_sdfg_call_args(args, fun.sdfg_program._lastargs[0])
            fun.fast_call()
        except IndexError:
            # First call, the SDFG is not initialized, so forward the call to `CompiledSDFG`
            # to properly initialize it. Later calls to this SDFG will be handled through
            # the `fast_call()` API.
            assert len(fun.sdfg_program._lastargs) == 0  # `fun.sdfg_program._lastargs` is empty
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(args)
            this_call_args = sdfg_callable.get_sdfg_args(
                fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            this_call_args |= {
                gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL: config.COLLECT_METRICS_LEVEL,
                gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME: collect_time_arg,
            }
            with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
                fun(**this_call_args)

        if collect_time:
            metric_source = metrics.get_current_source()
            assert metric_source is not None
            metric_source.metrics[metrics.COMPUTE_METRIC].add_sample(collect_time_arg[0].item())

    return decorated_program
