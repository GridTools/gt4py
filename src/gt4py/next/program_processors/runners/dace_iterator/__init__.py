import numpy as np

import gt4py.next.iterator.ir as itir
from gt4py.next.program_processors.processor_interface import program_executor
from gt4py.next.iterator.transforms import apply_common_transforms
from .itir_to_sdfg import ItirToSDFG
from gt4py.next.type_system import type_translation, type_specifications as ts
from gt4py.next.iterator.embedded import LocatedField
from typing import Any


def convert_arg(arg: Any):
    if isinstance(arg, LocatedField):
        return np.asarray(arg)
    return arg


@program_executor
def run_dace_iterator(program: itir.FencilDefinition, *args, **kwargs) -> None:
    offset_provider = kwargs["offset_provider"]
    arg_types = [type_translation.from_value(arg) for arg in args]

    program = apply_common_transforms(program, offset_provider=offset_provider, force_inline_lift=True)
    sdfg_gen = ItirToSDFG(param_types=arg_types, offset_provider=offset_provider)
    sdfg = sdfg_gen.visit(program)
    
    sdfg.view()

    regular_args = {
        name.id: convert_arg(arg)
        for name, arg in zip(program.params, args)
    }

    array_params = [param for param, type_ in zip(program.params, arg_types) if isinstance(type_, ts.FieldType)]
    array_args = [arg for arg in args if isinstance(arg, LocatedField)]
    shape_args = {
        str(sym): size
        for param, arg in zip(array_params, array_args)
        for sym, size in zip(sdfg.arrays[str(param.id)].shape, np.asarray(arg).shape)
    }

    sdfg(**regular_args, **shape_args)
