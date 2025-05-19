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
from gt4py.next import common, utils as gtx_utils
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners.dace import sdfg_callable, workflow as dace_worflow


def convert_args(
    inp: dace_worflow.compilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] else False

    # We use the callback function provided by the compiled program to update the SDFG arglist.
    update_sdfg_call_args = functools.partial(
        inp.update_sdfg_ctype_arglist, device, inp.sdfg_argtypes
    )

    def decorated_program(
        *args: Any,
        offset_provider: common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)

        if inp.implicit_domain:
            # Generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            args = (*args, *size_args)

        if not sdfg_program._lastargs:
            # First call, the SDFG is not intitalized, so forward the call to `CompiledSDFG`
            # to proper initilize it. Later calls to this SDFG will be handled through
            # the `fast_call()` API.
            flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
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
            # Initialization of `_lastargs` was done by the `CompiledSDFG` object,
            #  so we just update it with the current call arguments.
            update_sdfg_call_args(args, sdfg_program._lastargs[0])
            return inp.fast_call()

    return decorated_program
