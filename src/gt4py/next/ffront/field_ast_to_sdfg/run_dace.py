# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from typing import Any, Sequence

import dace
import numpy as np
from dace.transformation.dataflow import MapFusion

from gt4py.next.ffront import program_ast as past
from gt4py.next.ffront.field_ast_to_sdfg import past_to_sdfg
from gt4py.next.iterator.embedded import LocatedField


def convert_arg(arg: Any) -> Any:
    if isinstance(arg, LocatedField):
        return np.asarray(arg)
    else:
        return arg


def run_dace(
    program: past.Program, closure_vars: dict[str, Any], args: Sequence[Any], kwargs: dict[str, Any]
):
    program_lowering = past_to_sdfg.PastToSDFG(closure_vars)
    program_lowering.visit(program)
    sdfg: dace.SDFG = program_lowering.sdfg
    sdfg.save("gt4py_dace.sdfg")

    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapFusion)

    converted_args = {}

    for param, arg in zip(program.params, args):
        converted_args[str(param.id)] = convert_arg(arg)

    for dim, size in enumerate(converted_args[program.params[0].id].shape):
        converted_args[f"size{dim}"] = size

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="RelWithDebInfo")
        dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(**converted_args)
