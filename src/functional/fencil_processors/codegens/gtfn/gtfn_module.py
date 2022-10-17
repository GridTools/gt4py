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

from functional import common
from functional.fencil_processors import processor_interface as fpi  # fencil processor interface
from functional.fencil_processors.codegens.gtfn import gtfn_backend
from functional.fencil_processors.source_modules import cpp_gen, source_modules
from functional.iterator import ir as itir
from functional.common import DimensionKind

def get_param_description(
    name: str, obj: Any
) -> source_modules.ScalarParameter | source_modules.BufferParameter:
    view = np.asarray(obj)
    if view.ndim > 0:
        return source_modules.BufferParameter(
            name, tuple(dim.value if dim.kind != DimensionKind.LOCAL else dim.value+"Dim" for dim in obj.axes), view.dtype
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
        parameters = [
            get_param_description(fencil_param.id, obj)
            for obj, fencil_param in zip(args, fencil.params)
        ]
        for name, connectivity in kwargs["offset_provider"].items():
            if isinstance(connectivity, common.Connectivity):
                parameters.append(source_modules.ConnectivityParameter("__conn_"+name.lower(), connectivity.origin_axis.value, name))
            elif isinstance(connectivity, common.Dimension):
                pass
            else:
                raise ValueError(f"Expected offset provider `{name}` to be a "
                                 f"`Connectivity` or `Dimension`, but got "
                                 f"{type(connectivity).__name__}")
        function = source_modules.Function(fencil.id, tuple(parameters))

        connectivity_args = []
        for name, connectivity in kwargs["offset_provider"].items():
            if isinstance(connectivity, common.Connectivity):
                nbtbl = f"as_neighbor_table<generated::{connectivity.origin_axis.value}_t, " \
                        f"generated::{name}_t, {connectivity.max_neighbors}>(__conn_{name.lower()})"
                connectivity_args.append(f"gridtools::hymap::keys<generated::{name}_t>::make_values({nbtbl})")  # TODO std::forward, type and max_neighbors)
            elif isinstance(connectivity, common.Dimension):
                pass
            else:
                raise ValueError(f"Expected offset provider `{name}` to be a "
                                 f"`Connectivity` or `Dimension`, but got "
                                 f"{type(connectivity).__name__}")
        rendered_connectivity_args = ", ".join(connectivity_args)

        import eve.trees
        import eve.utils
        scalar_parameters = eve.trees.pre_walk_values(eve.utils.XIterable(fencil.closures).getattr("inputs").to_list()).if_isinstance(itir.SymRef).getattr("id").map(str).to_list()

        parameter_args = ["gridtools::fn::backend::naive{}"]
        for p in parameters:
            if isinstance(p, (source_modules.ScalarParameter, source_modules.BufferParameter)):
                if isinstance(p, source_modules.ScalarParameter) and p.name in scalar_parameters:
                    parameter_args.append(f"gridtools::stencil::global_parameter({p.name})")
                else:
                    parameter_args.append(p.name)

        rendered_parameter_args = ", ".join(parameter_args)
        #decl_body = f"double* bla=&__sym_2; return generated::{function.name}()({rendered_params});"
        decl_body = f"return generated::{function.name}({rendered_connectivity_args})({rendered_parameter_args});"
        decl_src = cpp_gen.render_function_declaration(function, body=decl_body)
        stencil_src = gtfn_backend.generate(fencil, **kwargs)
        source_code = source_modules.format_source(
            self.language_settings,
            f"""
            #include <gridtools/fn/backend/naive.hpp>
            #include <gridtools/stencil/global_parameter.hpp>
            {stencil_src}
            using gridtools::fn::sid_neighbor_table::as_neighbor_table;
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
