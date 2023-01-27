import dace
import numpy as np

from functional.ffront import program_ast as past
from typing import Any, Sequence
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
    sdfg.save("/home/petiaccja/Work/ETHZ/tmp/gt4py_dace.sdfg")
    sdfg.view()

    converted_args = [convert_arg(arg) for arg in args]

    with dace.config.set_temporary("compiler", "build_type", value="Debug"):
        with dace.config.set_temporary("compiler", "cpu", "args", value="-O0"):
            sdfg(*converted_args)
