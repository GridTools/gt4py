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

from functional import common
from functional.fencil_processors import source_modules
from functional.fencil_processors.builders import cpp as cpp_callable
from functional.fencil_processors.codegens.gtfn import gtfn_backend, gtfn_module as gtfn_codegen
from functional.iterator import ir
from functional.iterator.processor_interface import fencil_executor


_dimension_kind_to_tag = {
    common.DimensionKind.HORIZONTAL: "gridtools::fn::unstructured::dim::horizontal",
    common.DimensionKind.VERTICAL: "gridtools::fn::unstructured::dim::vertical",
}  # TODO probably not the right place


# def get_arg_types(*args, grid_type) -> list[defs.ScalarParameter | defs.BufferParameter]:
#     def get_arg_type(arg):
#         view = numpy.array(arg)
#         if view.ndim > 0:
#             if grid_type == "unstructured":  # TODO introduce a common enum for grid_type
#                 return defs.BufferParameter(
#                     "", [_dimension_kind_to_tag[dim.kind] for dim in arg.axes], view.dtype.type
#                 )
#             else:
#                 return defs.BufferParameter(
#                     "", [f"generated::{dim.value}_t" for dim in arg.axes], view.dtype.type
#                 )  # TODO not the right place
#         else:
#             return defs.ScalarParameter("", type(arg))

#     return [get_arg_type(arg) for arg in args]


# def convert_args(*args) -> Sequence[Any]:
#     def convert_arg(arg):
#         view = numpy.asarray(arg)
#         if view.ndim > 0:
#             return memoryview(view)
#         else:
#             return arg

#     return [convert_arg(arg) for arg in args]


# def neighbortable_args(offset_provider):
#     return [c.tbl for c in offset_provider.values() if isinstance(c, Connectivity)]


# @fencil_executor
# def run_gtfn(itir: ir.FencilDefinition, *args, **kwargs):
#     assert "offset_provider" in kwargs
#     grid_type = gtfn_backend._guess_grid_type(**kwargs)
#     parameters = get_arg_types(*args, grid_type=grid_type)  # TODO cleanup handling of grid_type
#     for fparam, iparam in zip(parameters, itir.params):
#         fparam.name = iparam.id

#     for name, c in kwargs["offset_provider"].items():
#         if isinstance(c, Connectivity):
#             parameters.append(defs.ConnectivityParameter(name, name))
#     source_module = gtfn_codegen.create_source_module(itir, parameters, **kwargs)
#     wrapper = cpp_callable.create_callable(source_module, cache_strategy=Strategy.PERSISTENT)
#     wrapper(*convert_args(*args), *neighbortable_args(kwargs["offset_provider"]))


def get_param_description(
    name, obj, *, grid_type
) -> source_modules.ScalarParameter | source_modules.BufferParameter:
    view = numpy.asarray(obj)
    if view.ndim > 0:
        if grid_type == "unstructured":  # TODO introduce a common enum for grid_type
            return source_modules.BufferParameter(
                name, [_dimension_kind_to_tag[dim.kind] for dim in obj.axes], view.dtype
            )
        else:
            return source_modules.BufferParameter(
                name, [f"generated::{dim.value}_t" for dim in obj.axes], view.dtype
            )  # TODO not the right place
        # return source_modules.BufferParameter(name, [dim.value for dim in obj.axes], view.dtype)
    else:
        return source_modules.ScalarParameter(name, view.dtype)


def neighbortable_args(offset_provider):
    return [c.tbl for c in offset_provider.values() if isinstance(c, common.Connectivity)]


def convert_arg(arg) -> Any:
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

    assert "offset_provider" in kwargs
    grid_type = gtfn_backend.guess_grid_type(**kwargs)
    parameters = [
        get_param_description(itir_param.id, obj, grid_type=grid_type)
        for obj, itir_param in zip(args, itir.params)
    ]
    for name, c in kwargs["offset_provider"].items():
        if isinstance(c, common.Connectivity):
            parameters.append(source_modules.ConnectivityParameter(name, name))
    source_module = gtfn_codegen.create_source_module(itir, parameters, **kwargs)
    wrapper = cpp_callable.create_callable(source_module)
    wrapper(*[convert_arg(arg) for arg in args], *neighbortable_args(kwargs["offset_provider"]))
