from typing import Sequence, Any

import numpy

from . import defs
from .callable import cpp as cpp_callable
from .codegen import gtfn as gtfn_codegen
from functional.iterator import ir
from functional.iterator.backends.backend import register_backend


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


def run_gtfn(itir: ir.FencilDefinition, *args, **kwargs):
    parameters = get_arg_types(*args)
    for fparam, iparam in zip(parameters, itir.params):
        fparam.name = iparam.id
    source_module = gtfn_codegen.create_source_module(itir, parameters)
    wrapper = cpp_callable.create_callable(source_module)
    wrapper(*convert_args(*args))


register_backend("run_gtfn", run_gtfn)
