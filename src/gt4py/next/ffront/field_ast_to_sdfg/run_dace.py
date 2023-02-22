from typing import Any, Sequence

import dace
from dace.transformation.dataflow import MapFusion
import numpy as np

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

    # sdfg.view()

    converted_args = {}

    for param, arg in zip(program.params, args):
        converted_args[str(param.id)] = convert_arg(arg)

    for dim, size in enumerate(converted_args[program.params[0].id].shape):
        converted_args[f"size{dim}"] = size

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="RelWithDebInfo")
        # dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(**converted_args)
