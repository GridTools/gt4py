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


from typing import Sequence

from eve.codegen import format_source
from functional.fencil_processors import cpp, defs
from functional.fencil_processors.codegens.gtfn import gtfn_backend
from functional.iterator.ir import FencilDefinition


def create_source_module(
    itir: FencilDefinition,
    parameters: Sequence[defs.ScalarParameter | defs.BufferParameter],
    **kwargs,
) -> defs.SourceCodeModule:
    function = defs.Function(itir.id, parameters)

    rendered_params = ", ".join(["gridtools::fn::backend::naive{}", *[p.name for p in parameters]])
    decl_body = f"return generated::{function.name}(nullptr)({rendered_params});"
    decl_src = cpp.render_function_declaration(function, body=decl_body)
    stencil_src = gtfn_backend.generate(
        itir, grid_type=gtfn_backend._guess_grid_type(**kwargs), **kwargs
    )
    source_code = format_source(
        "cpp",
        f"""\
                                #include <gridtools/fn/backend/naive.hpp>
                                {stencil_src}
                                {decl_src}\
                                """,
        style="LLVM",
    )

    module = defs.SourceCodeModule(
        entry_point=function,
        library_deps=[
            defs.LibraryDependency("gridtools", "master"),
        ],
        source_code=source_code,
        language=cpp.language_id,
    )
    return module
