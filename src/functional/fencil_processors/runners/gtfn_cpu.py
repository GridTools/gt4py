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
from typing import Any, Callable, Final, Optional

import numpy as np

from functional.fencil_processors import pipeline, processor_interface as fpi
from functional.fencil_processors.builders import cache, otf_compiler
from functional.fencil_processors.builders.cpp import bindings, compiledb
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
    builder_factory: pipeline.OTFBuilderGenerator = compiledb.make_compiledb_factory()

    name: Optional[str] = None

    def __call__(self, fencil: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
        """
        Execute the iterator IR fencil with the provided arguments.

        The fencil is compiled to machine code with C++ as an intermediate step,
        so the first execution is expected to have a significant overhead, while subsequent
        calls are very fast. Only scalar and buffer arguments are supported currently.

        See ``FencilExecutorFunction`` for details.
        """

        def convert_args(fencil: Callable) -> Callable:
            def decorated_fencil(*args):
                return fencil(*[convert_arg(arg) for arg in args])

            return decorated_fencil

        def itir_to_src(inp: pipeline.OTFClosure) -> source_modules.SourceModule:
            return gtfn_module.GTFNSourceModuleGenerator(self.language_settings)(
                inp.entry_point, *inp.args, **inp.kwargs
            )

        def src_to_otf(inp: source_modules.SourceModule) -> source_modules.OTFSourceModule:
            return source_modules.OTFSourceModule(
                source_module=inp, bindings_module=bindings.create_bindings(inp)
            )

        otf_workflow: Final[pipeline.OTFWorkflow[pipeline.OTFClosure, Callable]] = (
            pipeline.OTFWorkflow(itir_to_src, src_to_otf)
            .add_step(
                otf_compiler.OnTheFlyCompiler(
                    cache_strategy=cache.Strategy.SESSION, builder_factory=self.builder_factory
                )
            )
            .add_step(convert_args)
        )

        otf_closure = pipeline.OTFClosure(fencil, args, kwargs)

        compiled_runner = otf_workflow(otf_closure)

        compiled_runner(*args)

    @property
    def __name__(self) -> str:
        return self.name or repr(self)


run_gtfn: Final[fpi.FencilProcessorProtocol[None, fpi.FencilExecutor]] = GTFNExecutor(
    name="run_gtfn"
)
