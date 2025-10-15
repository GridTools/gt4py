# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import dace
import numpy as np

from gt4py.next import common as gtx_common, config, metrics
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace.workflow import compilation as gtx_wfdcompilation


def convert_args(
    fun: gtx_wfdcompilation.CompiledDaceProgram,
) -> stages.CompiledProgram:
    # Retieve metrics level from GT4Py environment variable.
    collect_time = config.COLLECT_METRICS_LEVEL >= metrics.PERFORMANCE

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            result = fun(offset_provider, *args)

        if collect_time:
            if result is None:
                raise RuntimeError(
                    "Config 'COLLECT_METRICS_LEVEL' is set but the SDFG profiling"
                    " report was not found. This might indicate that the backend"
                    " is using a precompiled SDFG from persistent cache without"
                    " metrics instrumentation."
                )
            assert len(result) == 1
            assert isinstance(result[0], np.float64)
            metric_source = metrics.get_current_source()
            assert metric_source is not None
            metric_source.metrics[metrics.COMPUTE_METRIC].add_sample(result[0].item())

    return decorated_program
