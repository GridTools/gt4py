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
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.instrumentation import metrics
from gt4py.next.program_processors.runners.dace import sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


if TYPE_CHECKING:
    # Type-only: a top-level import would cycle with ``compilation``.
    from gt4py.next.program_processors.runners.dace.workflow.compilation import CompiledDaceProgram


class DaCeDecoratedProgram:
    """A compiled DaCe program wrapped as a GT4Py-callable ``ExecutableProgram``.

    Every call translates the GT4Py arguments through the binding function
    generated for the program and hands them to the SDFG via ``user_bind_call``.

    External workspace memory (when the SDFG uses
    ``TransientMemoryMode.EXTERNAL``) is installed onto the underlying
    ``CompiledDaceProgram`` via `set_external_workspace`; its lifetime is owned
    by the caller, not by this wrapper. It is handed to the SDFG on the first
    call, using that call's symbol values, since the required size is only known
    then. Keeping the buffer large enough for later calls, which may use
    different symbol values, is the caller's responsibility.
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
        # The external workspace is installed lazily, on the first call, because its
        # required size depends on the symbol values of that call.
        self._external_workspace_configured = False

    def __call__(
        self,
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        if self._fun.requires_external_workspace and not self._external_workspace_configured:
            self._configure_external_workspace(args, offset_provider)
            self._external_workspace_configured = True

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

    def _configure_external_workspace(
        self, args: tuple[Any, ...], offset_provider: gtx_common.OffsetProvider
    ) -> None:
        """Install the external workspace, using the symbol values of the current call.

        `configure_external_workspace()` needs the SDFG free symbols by name, which
        the binding function does not produce: it emits the positional `user_args`
        vector, from which DaCe infers the symbols on its own. So the SDFG argument
        mapping is built explicitly here. This is the slow path, but it only runs
        once per program, and only for programs with external transients.
        """
        flat_args = gtx_utils.flatten_nested_tuple(args)
        call_args = sdfg_callable.get_sdfg_args(
            self._fun.sdfg_program.sdfg, offset_provider, *flat_args, filter_args=False
        )
        call_args |= {
            gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL: metrics.get_current_level(),
            gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME: self._collect_time_arg,
        }
        self._fun.configure_external_workspace(**call_args)

    def set_external_workspace(self, external_workspace: gtx_wfdcommon.ExternalWorkspace) -> None:
        """Set the external workspace for the underlying compiled program.

        This method should be called before the first call to the program.
        """
        self._fun.external_workspace = external_workspace
