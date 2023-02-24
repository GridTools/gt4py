# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any

import dace
import numpy as np

import gt4py.next.iterator.ir as itir
from gt4py.next.iterator.embedded import LocatedField, NeighborTableOffsetProvider
from gt4py.next.iterator.transforms import apply_common_transforms
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.type_system import type_specifications as ts, type_translation
from .utility import connectivity_identifier

from .itir_to_sdfg import ItirToSDFG


def convert_arg(arg: Any):
    if isinstance(arg, LocatedField):
        return np.asarray(arg)
    return arg


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    offset_provider = kwargs["offset_provider"]
    arg_types = [type_translation.from_value(arg) for arg in args]

    program = apply_common_transforms(
        program, offset_provider=offset_provider, force_inline_lift=True
    )
    sdfg_gen = ItirToSDFG(param_types=arg_types, offset_provider=offset_provider)
    from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas

    program = InlineLambdas.apply(program, opcount_preserving=False, force_inline_lift=True)
    sdfg = sdfg_gen.visit(program)

    regular_args = {name.id: convert_arg(arg) for name, arg in zip(program.params, args)}

    neighbor_tables = [
        (offset, table)
        for offset, table in offset_provider.items()
        if isinstance(table, NeighborTableOffsetProvider)
    ]

    connectivity_args = {
        connectivity_identifier(offset): table.table for offset, table in neighbor_tables
    }
    connectivity_shape_args = {
        str(sym): size
        for offset, _ in neighbor_tables
        for sym, size in zip(
            sdfg.arrays[connectivity_identifier(offset)].shape, offset_provider[offset].table.shape
        )
    }

    array_params = [
        param for param, type_ in zip(program.params, arg_types) if isinstance(type_, ts.FieldType)
    ]
    array_args = [arg for arg in args if isinstance(arg, LocatedField)]
    shape_args = {
        str(sym): size
        for param, arg in zip(array_params, array_args)
        for sym, size in zip(sdfg.arrays[str(param.id)].shape, np.asarray(arg).shape)
    }
    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="Debug")
        dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(**regular_args, **shape_args, **connectivity_args, **connectivity_shape_args)
