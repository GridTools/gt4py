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
    use_field_canonical_representation: bool = False,
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

        if sdfg_program._lastargs:
            last_call_args = sdfg_program._lastargs[0]
            kwargs = dict(zip(sdfg.arg_names, flat_args, strict=True))
            for i, (arg_name, arg_desc) in enumerate(inp.sdfg_arglist):
                arg = kwargs.get(arg_name, None)
                if arg is None:
                    # connectivities are supposed not to change, and can therefore be omitted
                    assert gtx_dace_utils.is_connectivity_identifier(
                        arg_name, common.offset_provider_to_type(offset_provider)
                    ) or gtx_dace_utils.is_connectivity_symbol(
                        arg_name, common.offset_provider_to_type(offset_provider)
                    ), f"argument '{arg_name}' not found."
                elif (ndarray := getattr(arg, "ndarray", None)) is not None:
                    assert isinstance(arg_desc, dace.data.Array)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    assert field_utils.verify_device_field_type(arg, device)
                    last_call_args[i].value = (
                        ndarray.__cuda_array_interface__["data"][0]
                        if on_gpu
                        else ndarray.__array_interface__["data"][0]
                    )
                    # backfill dictionary of arguments with field shape and stride symbols
                    kwargs.update(sdfg_callable.get_field_domain_symbols(arg_name, arg.domain))
                    kwargs.update(sdfg_callable.get_field_stride_symbols(arg_desc, ndarray))
                else:
                    assert isinstance(arg_desc, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    actype = arg_desc.dtype.as_ctypes()
                    last_call_args[i] = actype(arg)

            return inp.fast_call()

        sdfg_args = sdfg_callable.get_sdfg_args(
            sdfg,
            offset_provider,
            *flat_args,
            filter_args=False,
            on_gpu=on_gpu,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
