# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.instrumentation import metrics
from gt4py.next.program_processors.runners.dace import sdfg_callable
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


if TYPE_CHECKING:
    # Type-only: a top-level import would cycle with ``compilation``.
    from gt4py.next.program_processors.runners.dace.workflow.compilation import CompiledDaceProgram


def _validate_external_sync_stream(
    stream: Any,
) -> None:
    """Validate that ``stream`` is a usable external synchronization stream.

    Args:
        stream: The object the user provided as external sync stream.

    Raises:
        TypeError: If ``stream`` is not a ``cupy.cuda.Stream``.
        ValueError: If the stream's device does not match the target device, or
            if the stream handle is invalid.
    """
    import cupy as cp

    if cp is None or not isinstance(stream, cp.cuda.Stream):
        raise TypeError("external_sync_stream must be a cupy.cuda.Stream.")

    current_device_id = cp.cuda.Device().id
    if stream.device_id != current_device_id:
        raise ValueError(
            f"external_sync_stream is on device {stream.device_id}, "
            f"but the current device is {current_device_id}."
        )

    result = cp.cuda.runtime.cudaStreamQuery(stream.ptr)
    if result not in (cp.cuda.runtime.cudaSuccess, cp.cuda.runtime.cudaErrorNotReady):
        raise ValueError(f"external_sync_stream failed cudaStreamQuery with error {result}.")


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
        self._collect_time = metrics.is_level_enabled(metrics.PERFORMANCE)
        self._collect_time_arg = np.array(
            [1], dtype=gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME_DTYPE.as_numpy_dtype()
        )
        # We use the callback function provided by the compiled program to update the SDFG arglist.
        self._device_type = device_type
        self._update_sdfg_call_args = functools.partial(
            fun.update_sdfg_ctype_arglist, device_type, fun.sdfg_argtypes
        )

    def __call__(
        self,
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> Any:
        if out is not None:
            args = (*args, out)

        try:
            # Not the first call.
            #  We will only update the argument vector  for the normal call.
            # NOTE: If this is the first time then we will generate an exception because
            #   `fun.csdfg_args` is `None`
            # TODO(phimuell, edopao): Think about refactor the code such that the update
            #   of the argument vector is a Method of the `CompiledDaceProgram`.
            self._update_sdfg_call_args(args, self._fun.csdfg_argv, offset_provider)  # type: ignore[arg-type]  # Will error out in first call.

        except TypeError:
            # First call. Construct the initial argument vector of the `CompiledDaceProgram`.
            assert self._fun.csdfg_argv is None and self._fun.csdfg_init_argv is None
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(args)
            this_call_args = sdfg_callable.get_sdfg_args(
                self._fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            this_call_args |= {
                gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL: metrics.get_current_level(),
                gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME: self._collect_time_arg,
            }
            self._fun.construct_arguments(**this_call_args)

        # Perform the call to the SDFG.
        self._fun.fast_call()

        if self._collect_time:
            metrics.add_sample_to_current_source(
                metrics.COMPUTE_METRIC, self._collect_time_arg[0].item()
            )

    def set_external_workspace(self, external_workspace: gtx_wfdcommon.ExternalWorkspace) -> None:
        """Set the external workspace for the underlying compiled program.

        This method should be called before the first call to the program.
        """
        self._fun.external_workspace = external_workspace

    def set_external_sync_stream(self, external_sync_stream: Any | None) -> None:
        """Set the external sync stream for the underlying compiled program.

        This method should be called before the first call to the program.
        """
        if external_sync_stream is None:
            return

        if self._device_type == core_defs.DeviceType.CPU:
            raise ValueError("Stream synchronization is not supported for CPU target.")

        _validate_external_sync_stream(external_sync_stream)
        self._fun.external_sync_stream = external_sync_stream
