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

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config, metrics, utils as gtx_utils
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners.dace import sdfg_callable, workflow as dace_worflow

from . import common as dace_common


def convert_args(
    fun: dace_worflow.compilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> stages.CompiledProgram:
    # We use the callback function provided by the compiled program to update the SDFG arglist.
    update_sdfg_call_args = functools.partial(
        fun.update_sdfg_ctype_arglist, device, fun.sdfg_argtypes
    )

    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)

        if fun.implicit_domain:
            # Generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            args = (*args, *size_args)

        if not fun.sdfg_program._lastargs:
            # First call, the SDFG is not intitalized, so forward the call to `CompiledSDFG`
            # to proper initilize it. Later calls to this SDFG will be handled through
            # the `fast_call()` API.
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
            this_call_args = sdfg_callable.get_sdfg_args(
                fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
                fun(**this_call_args)

        else:
            # Initialization of `_lastargs` was done by the `CompiledSDFG` object,
            #  so we just update it with the current call arguments.
            update_sdfg_call_args(args, fun.sdfg_program._lastargs[0])
            fun.fast_call()

        metric_collection = metrics.get_active_metric_collection()
        if (metric_collection is not None) and (
            config.COLLECT_METRICS_LEVEL >= metrics.PERFORMANCE
        ):
            # Observe that dace instrumentation adds runtime overhead:
            # DaCe writes an instrumentation report file for each SDFG run.
            with dace.config.temporary_config():
                # We need to set the cache folder and key config in order to retrieve
                # the SDFG report file.
                dace_common.set_dace_config(device_type=device)
                prof_report = fun.sdfg_program.sdfg.get_latest_report()
            if prof_report is None:
                raise RuntimeError(
                    "Config 'COLLECT_METRICS_LEVEL' is set but the SDFG profiling report was not found."
                    " This might indicate that the backend is using a precompiled SDFG from persistent cache"
                    " without instrumentation."
                )
            assert len(prof_report.events) == 1
            # The event name gets truncated in dace, so we only check that
            # it corresponds to the beginning of SDFG label.
            assert f"SDFG {fun.sdfg_program.sdfg.label}".startswith(prof_report.events[0].name)
            duration_secs = (
                prof_report.events[0].duration / 1e6
            )  # dace timer returns the duration in microseconds
            metric_collection.add_sample(metrics.COMPUTE_METRIC, duration_secs)

    return decorated_program
