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
from typing import Any, Final

import numpy as np

from functional.iterator import ir as itir
from functional.otf import languages, stages
from functional.otf.source import cpp_gen, source
from functional.program_processors import processor_interface as fpi  # fencil processor interface
from functional.program_processors.codegens.gtfn import gtfn_backend


def get_param_description(name: str, obj: Any) -> source.ScalarParameter | source.BufferParameter:
    view: np.ndarray = np.asarray(obj)
    if view.ndim > 0:
        return source.BufferParameter(name, tuple(dim.value for dim in obj.axes), view.dtype)
    else:
        return source.ScalarParameter(name, view.dtype)


@dataclasses.dataclass(frozen=True)
class GTFNSourceGenerator(fpi.ProgramSourceGenerator):
    language_settings: languages.LanguageWithHeaderFilesSettings = cpp_gen.CPP_DEFAULT

    def __call__(
        self,
        program: itir.FencilDefinition,
        *args,
        **kwargs,
    ) -> stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings]:
        """Generate GTFN C++ code from the ITIR definition."""
        parameters = tuple(
            get_param_description(program_param.id, obj)
            for obj, program_param in zip(args, program.params)
        )
        function = source.Function(program.id, parameters)

        rendered_params = ", ".join(
            ["gridtools::fn::backend::naive{}", *(p.name for p in parameters)]
        )
        decl_body = f"return generated::{function.name}()({rendered_params});"
        decl_src = cpp_gen.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(program, **kwargs)
        source_code = source.format_source(
            self.language_settings,
            f"""
            #include <gridtools/fn/backend/naive.hpp>
            {stencil_src}
            {decl_src}
            """.strip(),
        )

        module = stages.ProgramSource(
            entry_point=function,
            library_deps=(source.LibraryDependency("gridtools", "master"),),
            source_code=source_code,
            language=languages.Cpp,
            language_settings=self.language_settings,
        )
        return module


create_source_module: Final[
    fpi.ProgramProcessorProtocol[
        stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
        fpi.ProgramSourceGenerator,
    ]
] = GTFNSourceGenerator()
