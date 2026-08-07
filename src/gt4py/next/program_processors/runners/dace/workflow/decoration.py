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
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


if TYPE_CHECKING:
    # Type-only: a top-level import would cycle with ``compilation``.
    from gt4py.next.program_processors.runners.dace.workflow.compilation import CompiledDaceProgram


class DaCeDecoratedProgram:
    """A compiled DaCe program wrapped as a GT4Py-callable ``ExecutableProgram``.

    On the first call the full SDFG argument vector is constructed via
    ``CompiledDaceProgram.construct_arguments``; subsequent calls only update
    the argument vector in place through the binding function generated for the
    program. External workspace memory (when the SDFG uses
    ``TransientMemoryMode.EXTERNAL``) is installed onto the underlying
    ``CompiledDaceProgram`` before the first call via `set_external_workspace`;
    its lifetime is owned by the caller, not by this wrapper.
    """

    def __init__(
        self,
        fun: CompiledDaceProgram,
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    ) -> None:
        self._fun = fun
        # Retrieve metrics level from GT4Py environment variable.
        # TODO(phimuell): Should the check be in the call or at construction.
        self._collect_time = metrics.is_level_enabled(metrics.PERFORMANCE)
        self._collect_time_arg = np.array(
            [1], dtype=gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME_DTYPE.as_numpy_dtype()
        )

    def __call__(
        self,
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        processed_args, _ = self._fun.argument_preprocessing_function(
            args,
            offset_provider,
            metrics.get_current_level(),
            self._collect_time_arg,
        )
        self._fun.sdfg_program.user_bind_call(*processed_args)

        if self._collect_time:
            metrics.add_sample_to_current_source(
                metrics.COMPUTE_METRIC, self._collect_time_arg[0].item()
            )

    def set_external_workspace(self, external_workspace: gtx_wfdcommon.ExternalWorkspace) -> None:
        """Set the external workspace for the underlying compiled program.

        This method should be called before the first call to the program.
        """
        self._fun.external_workspace = external_workspace
