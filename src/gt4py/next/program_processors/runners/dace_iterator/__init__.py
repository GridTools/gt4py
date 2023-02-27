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

from typing import Any, Sequence, Mapping

import dace
import numpy as np

import gt4py.next.iterator.ir as itir
from gt4py.next.iterator.embedded import LocatedField, NeighborTableOffsetProvider
from gt4py.next.iterator.transforms import apply_common_transforms, inline_lambdas
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.type_system import type_translation
from .utility import connectivity_identifier, filter_neighbor_tables

from .itir_to_sdfg import ItirToSDFG


def convert_arg(arg: Any):
    if isinstance(arg, LocatedField):
        return np.asarray(arg)
    return arg


def preprocess_program(program: itir.FencilDefinition, offset_provider: Mapping[str, Any]):
    program = apply_common_transforms(
        program, offset_provider=offset_provider, force_inline_lift=True
    )
    program = inline_lambdas.InlineLambdas.apply(program, opcount_preserving=False, force_inline_lift=True)
    return program


def get_args(params: Sequence[itir.Sym], args: Sequence[Any]) -> dict[str, Any]:
    return {name.id: convert_arg(arg) for name, arg in zip(params, args)}


def get_connectivity_args(neighbor_tables: Sequence[tuple[str, NeighborTableOffsetProvider]]) -> dict[str, Any]:
    return {connectivity_identifier(offset): table.table for offset, table in neighbor_tables}


def get_shape_args(arrays: Mapping[str, dace.data.Array], args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(sym): size
        for name, value in args.items()
        for sym, size in zip(
            arrays[name].shape, value.shape
        )
    }


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    offset_provider = kwargs["offset_provider"]
    neighbor_tables = filter_neighbor_tables(offset_provider)

    program = preprocess_program(program, offset_provider)
    arg_types = [type_translation.from_value(arg) for arg in args]
    sdfg_genenerator = ItirToSDFG(param_types=arg_types, offset_provider=offset_provider)
    sdfg = sdfg_genenerator.visit(program)

    call_args = get_args(program.params, args)
    call_conn_args = get_connectivity_args(neighbor_tables)
    call_shapes = get_shape_args(sdfg.arrays, {n: v for n, v in call_args.items() if hasattr(v, "shape")})
    call_conn_shapes = get_shape_args(sdfg.arrays, call_conn_args)

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="Debug")
        dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(**call_args, **call_conn_args, **call_shapes, **call_conn_shapes)
