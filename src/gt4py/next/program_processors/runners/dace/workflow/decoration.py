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
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, utils as gtx_utils
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
            kwargs = dict(zip(sdfg.arg_names, flat_args, strict=True))
            kwargs.update(sdfg_callable.get_sdfg_conn_args(sdfg, offset_provider, on_gpu))

            use_fast_call = True
            last_call_args = sdfg_program._lastargs[0]
            # The scalar arguments should be overridden with the new value; for field arguments,
            # the data pointer should remain the same otherwise fast_call cannot be used and
            # the arguments list has to be reconstructed.
            for i, (arg_name, arg_type) in enumerate(inp.sdfg_arglist):
                if isinstance(arg_type, dace.data.Array):
                    assert arg_name in kwargs, f"argument '{arg_name}' not found."
                    data_ptr = get_array_interface_ptr(kwargs[arg_name], arg_type.storage)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    if last_call_args[i].value != data_ptr:
                        use_fast_call = False
                        break
                else:
                    assert isinstance(arg_type, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    if arg_name in kwargs:
                        # override the scalar value used in previous program call
                        actype = arg_type.dtype.as_ctypes()
                        last_call_args[i] = actype(kwargs[arg_name])
                    else:
                        # shape and strides of arrays are supposed not to change, and can therefore be omitted
                        assert gtx_dace_utils.is_field_symbol(
                            arg_name
                        ), f"argument '{arg_name}' not found."

            if use_fast_call:
                return inp.fast_call()

        sdfg_args = sdfg_callable.get_sdfg_args(
            sdfg,
            offset_provider,
            *flat_args,
            check_args=False,
            on_gpu=on_gpu,
            use_field_canonical_representation=use_field_canonical_representation,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
