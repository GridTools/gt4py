import functional.iterator.ir as itir
from eve import codegen
from functional.iterator.backends import backend
from functional.iterator.backends.gtfn.codegen import gtfn_codegen
from functional.iterator.backends.gtfn.itir_to_gtfn_ir import GTFN_lowering
from functional.iterator.transforms.common import apply_common_transforms


def generate(program: itir.Program, **kwargs) -> str:
    transformed = program
    transformed = apply_common_transforms(
        program,
        use_tmps=kwargs.get("use_tmps", False),
        offset_provider=kwargs.get("offset_provider", None),
        grid_type=kwargs.get("grid_type", None),
    )
    gtfn_ir = GTFN_lowering().visit(transformed)
    generated_code = gtfn_codegen.apply(gtfn_ir, **kwargs)
    formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
    return formatted_code


backend.register_backend("gtfn", lambda prog, *args, **kwargs: print(generate(prog, **kwargs)))
