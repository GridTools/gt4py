# GT4Py Project - GridTools Framework
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


from typing import Any, Sequence

import numpy

from functional.fencil_processors import source_modules
from functional.fencil_processors.builders import cpp as cpp_callable
from functional.fencil_processors.codegens.gtfn import gtfn_module as gtfn_codegen
from functional.iterator import ir
from functional.iterator.processor_interface import fencil_executor


def get_arg_types(
    *args,
) -> Sequence[source_modules.ScalarParameter | source_modules.BufferParameter]:
    def get_arg_type(arg):
        view = numpy.array(arg)
        if view.ndim > 0:
            return source_modules.BufferParameter(
                "", [dim.value for dim in arg.axes], view.dtype.type
            )
        else:
            return source_modules.ScalarParameter("", type(arg))

    return [get_arg_type(arg) for arg in args]


def convert_args(*args) -> Sequence[Any]:
    def convert_arg(arg):
        view = numpy.asarray(arg)
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
