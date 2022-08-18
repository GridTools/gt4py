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


from typing import Any

import numpy

from functional.fencil_processors.builders import cpp as cpp_callable
from functional.fencil_processors.codegens.gtfn import gtfn_module as gtfn_codegen
from functional.fencil_processors.processor_interface import fencil_executor
from functional.iterator import ir


def convert_arg(arg: Any) -> Any:
    view = numpy.asarray(arg)
    if view.ndim > 0:
        return memoryview(view)
    else:
        return arg


# TODO(ricoh): change style to declarative pipeline
@fencil_executor
def run_gtfn(itir: ir.FencilDefinition, *args, **kwargs):
    """
    Execute the iterator IR fencil with the provided arguments.

    The fencil is compiled to machine code with C++ as an intermediate step,
    so the first execution is expected to have a significant overhead, while subsequent
    calls are very fast. Only scalar and buffer arguments are supported currently.

    See ``FencilExecutorFunction`` for details.
    """
    source_module = gtfn_codegen.create_source_module(itir, *args, **kwargs)
    wrapper = cpp_callable.create_callable(source_module)
    wrapper(*[convert_arg(arg) for arg in args])
