# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
from typing import Any, Sequence

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import common, field_utils, utils as gtx_utils
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners.dace import (
    sdfg_callable,
    utils as gtx_dace_utils,
    workflow as dace_worflow,
)


def convert_args(
    inp: dace_worflow.compilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    assume_immutable_offset_provider: bool = True,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] else False

    def decorated_program(
        *args: Any,
        offset_provider: common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
        if inp.implicit_domain:
            # generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            flat_size_args: Sequence[int] = gtx_utils.flatten_nested_tuple(tuple(size_args))
            flat_args = (*flat_args, *flat_size_args)

        if not sdfg_program._lastargs:
            # First call, the SDFG is not intitalized, so forward the call to `CompiledSDFG`
            # to proper initilize it. Later calls to this SDFG will be handled through
            # the `fast_call()` API.
            this_call_args = sdfg_callable.get_sdfg_args(
                sdfg,
                offset_provider,
                *flat_args,
                filter_args=False,
                on_gpu=on_gpu,
            )
            with dace.config.temporary_config():
                dace.config.Config.set("compiler", "allow_view_arguments", value=True)
                return inp(**this_call_args)

        else:
            # Initialization of `_lastargs` was done by the `CompiledSDFG` object.
            last_call_args = sdfg_program._lastargs[0]
            this_call_args = dict(zip(sdfg.arg_names, flat_args, strict=True))
            if not assume_immutable_offset_provider:
                this_call_args.update(
                    sdfg_callable.get_sdfg_conn_args(sdfg, offset_provider, on_gpu)
                )
            # The loop below will traverse the `sdfg_arglist`, whose order reflects
            # the DaCe calling convention: first the array arguments, then the scalar
            # arguments. We exploit this knowledge to update `this_call_args` with
            # the shape and stride symbols of array arguments, when they are visited.
            # The corresponding scalar arguments are visited later in `sdfg_arglist`.
            for i, (arg_name, arg_desc) in enumerate(inp.sdfg_arglist):
                arg = this_call_args.get(arg_name, None)
                if arg is None:
                    # In case of immutable offset provider, the connectivity arrays
                    # and the associate shape and stride symbols can be omitted.
                    assert assume_immutable_offset_provider and (
                        gtx_dace_utils.is_connectivity_identifier(
                            arg_name, common.offset_provider_to_type(offset_provider)
                        )
                        or gtx_dace_utils.is_connectivity_symbol(
                            arg_name, common.offset_provider_to_type(offset_provider)
                        )
                    ), f"argument '{arg_name}' not found."
                elif (ndarray := getattr(arg, "ndarray", None)) is None:
                    assert isinstance(arg_desc, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    actype = arg_desc.dtype.as_ctypes()
                    last_call_args[i] = actype(arg)
                else:
                    assert isinstance(arg_desc, dace.data.Array)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    assert field_utils.verify_device_field_type(arg, device)
                    last_call_args[i].value = (
                        ndarray.__cuda_array_interface__["data"][0]
                        if on_gpu
                        else ndarray.__array_interface__["data"][0]
                    )
                    # When we find an array we update the `this_call_args` map
                    # with the shape and stride symbols that are associated to it.
                    # Note that `inp.sdfg_arglist` was constructed from an ordered
                    # dictionary whose order reflects the DaCe calling convention:
                    # first the array arguments, then the scalar arguments.
                    # Thus, when we enter the branch above for scalar arguments
                    # we know that all arrays have been processed and all their
                    # associated symbols have been added to `this_call_args`.
                    this_call_args.update(
                        sdfg_callable.get_field_domain_symbols(arg_name, arg.domain)
                    )
                    this_call_args.update(sdfg_callable.get_array_stride_symbols(arg_desc, ndarray))
            else:
                return inp.fast_call()

    return decorated_program
