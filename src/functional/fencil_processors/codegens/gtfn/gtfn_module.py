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
