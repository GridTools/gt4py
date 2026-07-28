# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args, sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    nanobinings as gtx_dace_nano,
)


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

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        if False:
            # Simply forward the call.
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(args)

            this_call_args = sdfg_callable.get_sdfg_args(
                fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            this_call_args |= {
                gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL: metrics.get_current_level(),
                gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME: collect_time_arg,
            }
            fun.sdfg_program(**this_call_args)
        else:
            args = gtx_dace_nano.convert_arg(args)

            # Works becauuse of constant prefix, but it is very fragile.
            args = args + tuple(
                offset_provider[pn].ndarray
                for pn in sorted(
                    pn
                    for pn in offset_provider.keys()
                    if gtx_dace_args.connectivity_identifier(pn) in fun.sdfg_program.sdfg.user_args
                )
            )

            if gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME in fun.sdfg_program.sdfg.arrays:
                args = (*args, metrics.get_current_level(), collect_time_arg)

            fun.sdfg_program.user_bind_call(*args)

        if collect_time:
            metrics.add_sample_to_current_source(metrics.COMPUTE_METRIC, collect_time_arg[0].item())

    return decorated_program
