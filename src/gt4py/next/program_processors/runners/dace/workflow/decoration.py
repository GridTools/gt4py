# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
from typing import Any, Callable, Sequence

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, config, field_utils, metrics, utils as gtx_utils
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners.dace import (
    sdfg_callable,
    utils as gtx_dace_utils,
    workflow as dace_worflow,
)


def update_sdfg_args(
    fun: dace_worflow.compilation.CompiledDaceProgram,
    flat_args: Sequence[Any],
    device: core_defs.DeviceType,
    offset_provider: gtx_common.OffsetProvider,
    assume_immutable_offset_provider: bool,
) -> None:
    sdfg = fun.sdfg_program.sdfg
    sdfg_program = fun.sdfg_program

    last_call_args = sdfg_program._lastargs[0]
    this_call_args = dict(zip(sdfg.arg_names, flat_args, strict=True))
    if not assume_immutable_offset_provider:
        this_call_args.update(sdfg_callable.get_sdfg_conn_args(sdfg, offset_provider))
        # The loop below will traverse the `sdfg_arglist`, whose order reflects
        # the DaCe calling convention: first the array arguments, then the scalar
        # arguments. We exploit this knowledge to update `this_call_args` with
        # the shape and stride symbols of array arguments, when they are visited.
        # The corresponding scalar arguments are visited later in `sdfg_arglist`.
    for i, (arg_name, arg_desc) in enumerate(fun.sdfg_arglist):
        arg = this_call_args.get(arg_name, None)
        if arg is None:
            if assume_immutable_offset_provider and (
                gtx_dace_utils.is_connectivity_identifier(
                    arg_name, gtx_common.offset_provider_to_type(offset_provider)
                )
                or gtx_dace_utils.is_connectivity_symbol(
                    arg_name, gtx_common.offset_provider_to_type(offset_provider)
                )
            ):
                # In case of immutable offset provider, the connectivity arrays
                # and the associate shape and stride symbols can be omitted.
                pass
            else:
                raise ValueError("argument '{arg_name}' not found.")

        elif (ndarray := getattr(arg, "ndarray", None)) is not None:
            assert isinstance(arg_desc, dace.data.Array)
            assert isinstance(last_call_args[i], ctypes.c_void_p)
            assert field_utils.verify_device_field_type(arg, device)
            last_call_args[i].value = arg.data_ptr()
            # When we find an array we update `this_call_args` with the
            # shape and stride symbols that are associated to it.
            # Note that `sdfg_arglist` was constructed from an ordered
            # dictionary whose order reflects the DaCe calling convention:
            # first the array arguments, then the scalar arguments.
            # Thus, when we enter the branch below for scalar arguments
            # we know that all arrays have been processed and all their
            # associated symbols have been added to `this_call_args`.
            this_call_args.update(sdfg_callable.get_field_domain_symbols(arg_name, arg.domain))
            this_call_args.update(sdfg_callable.get_array_stride_symbols(arg_desc, ndarray))

        else:
            # As outlined above, because of how `sdfg_arglist` is constructed,
            # all arrays will be visited first, thus all scalars that are
            # associated with the shape and stride symbols have been updated
            # in `this_call_args` and it is thus safe to use them.
            assert isinstance(arg_desc, dace.data.Scalar)
            assert isinstance(last_call_args[i], ctypes._SimpleCData)
            actype = arg_desc.dtype.as_ctypes()
            last_call_args[i] = actype(arg)
    #
    # End looping over `sdfg_arglist`: the arguments have been updated on `CompiledSDFG` object.


def inject_timer(
    name: str, sdfg: dace.SDFG
) -> Callable[[stages.CompiledProgram], stages.CompiledProgram]:
    def outer(fun: stages.CompiledProgram) -> stages.CompiledProgram:
        if not config.COLLECT_METRICS:
            return fun

        def inner(*args: Any, **kwargs: Any) -> None:
            fun(*args, **kwargs)
            # Observe that dace instrumentation adds runtime overhead:
            #  for each SDFG run, dace saves the instrumentation report to a file.
            sdfg_events = sdfg.get_latest_report().events
            assert len(sdfg_events) == 1
            assert sdfg_events[0].name == f"SDFG {name}"
            duration_secs = (
                sdfg_events[0].duration / 1e6
            )  # dace timer returns the duration in microseconds
            metrics.global_metric_container[name][metrics.CPP].append(duration_secs)

        return inner

    return outer


def convert_args(
    fun: dace_worflow.compilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    assume_immutable_offset_provider: bool = True,
) -> stages.CompiledProgram:
    @inject_timer(fun.name, fun.sdfg_program.sdfg)
    def decorated_program(
        *args: Any,
        offset_provider: gtx_common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
        if fun.implicit_domain:
            # Generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            flat_size_args: Sequence[int] = gtx_utils.flatten_nested_tuple(tuple(size_args))
            flat_args = (*flat_args, *flat_size_args)

        if not fun.sdfg_program._lastargs:
            # First call, the SDFG is not intitalized, so forward the call to `CompiledSDFG`
            # to proper initilize it. Later calls to this SDFG will be handled through
            # the `fast_call()` API.
            this_call_args = sdfg_callable.get_sdfg_args(
                fun.sdfg_program.sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
            )
            with dace.config.temporary_config():
                dace.config.Config.set("compiler", "allow_view_arguments", value=True)
                return fun(**this_call_args)

        else:
            # Initialization of `_lastargs` was done by the `CompiledSDFG` object,
            #  so we just update it with the current call arguments.
            update_sdfg_args(
                fun, flat_args, device, offset_provider, assume_immutable_offset_provider
            )
            return fun.fast_call()

    return decorated_program
