from typing import Any

import functional.iterator.ir as itir
from eve import codegen
from functional.fencil_processors.gtfn.codegen import GTFNCodegen
from functional.fencil_processors.gtfn.itir_to_gtfn_ir import GTFN_lowering
from functional.iterator.embedded import NeighborTableOffsetProvider
from functional.iterator.processor_interface import fencil_formatter
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.pass_manager import apply_common_transforms


def generate(program: itir.FencilDefinition, *, grid_type: str, **kwargs: Any) -> str:
    transformed = program
    offset_provider = kwargs.get("offset_provider")
    transformed = apply_common_transforms(
        program,
        lift_mode=kwargs.get("lift_mode"),
        offset_provider=offset_provider,
        unroll_reduce=True,
    )
    transformed = EtaReduction().visit(transformed)
    gtfn_ir = GTFN_lowering().visit(
        transformed, grid_type=grid_type, offset_provider=offset_provider
    )
    generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def _guess_grid_type(**kwargs):
    assert "offset_provider" in kwargs
    return (
        "unstructured"
        if any(isinstance(o, NeighborTableOffsetProvider) for o in kwargs["offset_provider"])
        else "cartesian"
    )


@fencil_formatter
def format_sourcecode(fencil: itir.FencilDefinition, *arg, **kwargs) -> str:
    return generate(fencil, grid_type=_guess_grid_type(**kwargs), **kwargs)
