from typing import Any

import functional.iterator.ir as itir
from eve import codegen
from functional.iterator.backends.backend import register_backend
from functional.iterator.backends.gtfn.codegen import GTFNCodegen
from functional.iterator.backends.gtfn.itir_to_gtfn_ir import GTFN_lowering
from functional.iterator.embedded import NeighborTableOffsetProvider
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.pass_manager import apply_common_transforms


def generate(program: itir.FencilDefinition, *, grid_type: str, **kwargs: Any) -> str:
    transformed = program
    transformed = apply_common_transforms(
        program,
        lift_mode=kwargs.get("lift_mode"),
        offset_provider=kwargs.get("offset_provider", None),
        unroll_reduce=True,
    )
    transformed = EtaReduction().visit(transformed)
    gtfn_ir = GTFN_lowering().visit(transformed, grid_type=grid_type)
    generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def _guess_grid_type(**kwargs):
    assert "offset_provider" in kwargs
    return (
        "unstructured"
        if any(isinstance(o, NeighborTableOffsetProvider) for o in kwargs["offset_provider"])
        else "cartesian"
    )


register_backend(
    "gtfn",
    lambda prog, *args, **kwargs: print(
        generate(prog, grid_type=_guess_grid_type(**kwargs), **kwargs)
    ),
)
