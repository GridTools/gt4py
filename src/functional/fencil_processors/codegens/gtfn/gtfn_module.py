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

from functional.fencil_processors import processor_interface as fpi  # fencil processor interface
from functional.fencil_processors.codegens.gtfn import gtfn_backend
from functional.fencil_processors.source_modules import cpp_gen, source_modules
from functional.iterator import ir as itir


def get_param_description(
    name: str, obj: Any
) -> source_modules.ScalarParameter | source_modules.BufferParameter:
    view = np.asarray(obj)
    if view.ndim > 0:
        return source_modules.BufferParameter(
            name, tuple(dim.value for dim in obj.axes), view.dtype
        )
    else:
        return source_modules.ScalarParameter(name, view.dtype)


@dataclasses.dataclass(frozen=True)
class GTFNSourceModuleGenerator(fpi.FencilSourceModuleGenerator):
    language_settings: source_modules.LanguageWithHeaderFilesSettings = cpp_gen.CPP_DEFAULT

    def __call__(
        self,
        fencil: itir.FencilDefinition,
        *args,
        **kwargs,
    ) -> source_modules.SourceModule[
        source_modules.Cpp, source_modules.LanguageWithHeaderFilesSettings
    ]:
        """Generate GTFN C++ code from the ITIR definition."""
        parameters = tuple(
            get_param_description(fencil_param.id, obj)
            for obj, fencil_param in zip(args, fencil.params)
        )
        function = source_modules.Function(fencil.id, parameters)

        rendered_params = ", ".join(
            ["gridtools::fn::backend::naive{}", *(p.name for p in parameters)]
        )
        decl_body = f"return generated::{function.name}()({rendered_params});"
        decl_src = cpp_gen.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(fencil, **kwargs)
        source_code = source_modules.format_source(
            self.language_settings,
            f"""
            #include <gridtools/fn/backend/naive.hpp>
            {stencil_src}
            {decl_src}
            """.strip(),
        )

        module = source_modules.SourceModule(
            entry_point=function,
            library_deps=(source_modules.LibraryDependency("gridtools", "master"),),
            source_code=source_code,
            language=source_modules.Cpp,
            language_settings=self.language_settings,
        )
        return module


create_source_module: Final[
    fpi.FencilProcessorProtocol[
        source_modules.SourceModule[
            source_modules.Cpp, source_modules.LanguageWithHeaderFilesSettings
        ],
        fpi.FencilSourceModuleGenerator,
    ]
] = GTFNSourceModuleGenerator()
