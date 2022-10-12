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


import dataclasses
from typing import Any, Final, Optional

import numpy as np

from functional.fencil_processors import processor_interface as fpi  # fencil processor interface
from functional.fencil_processors.builders import cache
from functional.fencil_processors.builders.cpp import bindings, build
from functional.fencil_processors.codegens.gtfn import gtfn_module
from functional.fencil_processors.source_modules import cpp_gen, source_modules
from functional.iterator import ir as itir


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if hasattr(arg, "__array__"):
        return np.asarray(arg)
    else:
        return arg


@dataclasses.dataclass(frozen=True)
class GTFNExecutor(fpi.FencilExecutor):
    language_settings: source_modules.LanguageWithHeaderFilesSettings = cpp_gen.CPP_DEFAULT

    name: Optional[str] = None

    def __call__(self, fencil: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        """
        Execute the iterator IR fencil with the provided arguments.

        The fencil is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``FencilExecutorFunction`` for details.
        """
        # TODO(ricoh): a pipeline runner might enhance readability as well as discourage
        #  custom logic between steps.
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
    def __name__(self) -> str:
        return self.name or repr(self)


run_gtfn: Final[fpi.FencilProcessorProtocol[None, fpi.FencilExecutor]] = GTFNExecutor(
    name="run_gtfn"
)
