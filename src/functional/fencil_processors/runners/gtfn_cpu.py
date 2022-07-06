from typing import Any, Sequence

import numpy

from functional.fencil_processors import defs
from functional.fencil_processors.callables import cpp as cpp_callable
from functional.fencil_processors.codegens.gtfn import gtfn_module as gtfn_codegen
from functional.iterator import ir
from functional.iterator.processor_interface import fencil_executor


def get_arg_types(*args) -> Sequence[defs.ScalarParameter | defs.BufferParameter]:
    def get_arg_type(arg):
        view = numpy.array(arg)
        if view.ndim > 0:
            return defs.BufferParameter("", view.ndim, view.dtype.type)
        else:
            return defs.ScalarParameter("", type(arg))

    return [get_arg_type(arg) for arg in args]


def convert_args(*args) -> Sequence[Any]:
    def convert_arg(arg):
        view = numpy.array(arg)
        if view.ndim > 0:
            return memoryview(view)
        else:
            return arg

    return [convert_arg(arg) for arg in args]


@fencil_executor
def run_gtfn(itir: ir.FencilDefinition, *args, **kwargs):
    parameters = get_arg_types(*args)
    for fparam, iparam in zip(parameters, itir.params):
        fparam.name = iparam.id
    source_module = gtfn_codegen.create_source_module(itir, parameters, **kwargs)
    wrapper = cpp_callable.create_callable(source_module)
    wrapper(*convert_args(*args))
