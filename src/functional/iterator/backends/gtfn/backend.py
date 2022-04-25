import functional.iterator.ir as itir
from eve import codegen
from eve.utils import UIDs
from functional.iterator.backends import backend
from functional.iterator.backends.gtfn.codegen import gtfn_codegen
from functional.iterator.backends.gtfn.itir_to_gtfn_ir import GTFN_lowering
from functional.iterator.transforms.common import add_fundef, replace_node
from functional.iterator.transforms.extract_function import extract_function
from functional.iterator.transforms.pass_manager import apply_common_transforms


def extract_fundefs_from_closures(program: itir.FencilDefinition) -> itir.FencilDefinition:
    inlined_stencils = (
        program.iter_tree()
        .if_isinstance(itir.StencilClosure)
        .getattr("stencil")
        .if_not_isinstance(itir.SymRef)
    )

    for stencil in inlined_stencils:
        ref, fundef = extract_function(stencil, f"{program.id}_stencil_{UIDs.sequential_id()}")
        program = add_fundef(program, fundef)
        program = replace_node(program, stencil, ref)

    return program


def generate(program: itir.FencilDefinition, *, grid_type: str, **kwargs) -> str:
    transformed = program
    transformed = apply_common_transforms(
        program,
        use_tmps=kwargs.get("use_tmps", False),
        offset_provider=kwargs.get("offset_provider", None),
        grid_type=grid_type,
    )
    transformed = extract_fundefs_from_closures(transformed)
    gtfn_ir = GTFN_lowering().visit(transformed, grid_type=grid_type)
    generated_code = gtfn_codegen.apply(gtfn_ir, **kwargs)
    formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
    return formatted_code


backend.register_backend("gtfn", lambda prog, *args, **kwargs: print(generate(prog, **kwargs)))
