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


from dataclasses import dataclass, field
from typing import Any, Optional

import numpy

from functional.fencil_processors import processor_interface as fpi  # fencil processor interface
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import bindings, build
from functional.fencil_processors.codegens.gtfn import gtfn_module
from functional.fencil_processors.source_modules import cpp_gen
from functional.iterator import ir as itir


def convert_arg(arg: Any) -> Any:
    view = numpy.asarray(arg)
    if view.ndim > 0:
        return memoryview(view)  # type: ignore[arg-type] # mypy seems unaware that ndarray is compatible with buffer protocol
    else:
        return arg


@dataclass(frozen=True)
class GTFNExecutor(fpi.FencilExecutor):
    language_settings: cpp_gen.CppLanguage = field(default=cpp_gen.CPP_DEFAULT)
    name: Optional[str] = None

    def __call__(self, fencil: itir.FencilDefinition, *args, **kwargs):
        """
        Execute the iterator IR fencil with the provided arguments.

        The fencil is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``FencilExecutorFunction`` for details.
        """
        return build.CMakeProject(
            source_module=(
                source_module := gtfn_module.GTFNSourceModuleGenerator(self.language_settings)(
                    fencil, *args, **kwargs
                )
            ),
            bindings_module=bindings.create_bindings(source_module),
            cache_strategy=cache.Strategy.SESSION,
        ).get_implementation()(*[convert_arg(arg) for arg in args])

    @property
    def __name__(self):
        return self.name or repr(self)


run_gtfn: fpi.FencilProcessorProtocol[None, fpi.FencilExecutor] = GTFNExecutor(name="run_gtfn")
