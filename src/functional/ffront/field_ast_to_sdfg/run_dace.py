from typing import Any, Sequence

import dace
import numpy as np

from functional.ffront import program_ast as past
from functional.ffront.field_ast_to_sdfg import past_to_sdfg
from functional.iterator.embedded import LocatedField


def convert_arg(arg: Any) -> Any:
    if isinstance(arg, LocatedField):
        return np.asarray(arg)
    else:
        return arg


def run_dace(
    program: past.Program, closure_vars: dict[str, Any], args: Sequence[Any], kwargs: dict[str, Any]
):
    program_lowering = past_to_sdfg.PastToSDFG()
    program_lowering.visit(program)
    sdfg: dace.SDFG = program_lowering.sdfg
    sdfg.save("gt4py_dace.sdfg")
    sdfg.view()

    converted_args = [convert_arg(arg) for arg in args]

    shape_and_stride_symbols = dict()
    for param, arg in zip(program.params, converted_args):
        shape_and_stride_symbols[str(param.id)] = arg
        for i, value in enumerate(arg.shape):
            shape_and_stride_symbols[f"_size_{param.id}_{i}"] = value
        for i, value in enumerate(arg.strides):
            shape_and_stride_symbols[f"_stride_{param.id}_{i}"] = value // arg.dtype.itemsize

    with dace.config.temporary_config():
        dace.config.Config.set("compiler", "build_type", value="Debug")
        dace.config.Config.set("compiler", "cpu", "args", value="-O0")
        dace.config.Config.set("frontend", "check_args", value=True)
        sdfg(**shape_and_stride_symbols)
