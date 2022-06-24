from typing import Any, cast

import functional.iterator.ir as itir
from eve import codegen
from eve.utils import UIDs
from functional.fencil_processors.gtfn.codegen import GTFNCodegen
from functional.fencil_processors.gtfn.itir_to_gtfn_ir import GTFN_lowering
from functional.iterator.embedded import NeighborTableOffsetProvider
from functional.iterator.processor_interface import fencil_formatter
from functional.iterator.transforms.common import add_fundefs, replace_nodes
from functional.iterator.transforms.extract_function import extract_function
from functional.iterator.transforms.pass_manager import apply_common_transforms


def extract_fundefs_from_closures(program: itir.FencilDefinition) -> itir.FencilDefinition:
    # TODO this would not work if the SymRef is a ref to a builtin, e.g. `deref`.
    # We should adapt this filter and add support for extracting builtins in `extract_function`,
    # which requires type information for the builtins.
    inlined_stencils = (
        program.pre_walk_values()
        .if_isinstance(itir.StencilClosure)
        .getattr("stencil")
        .if_not_isinstance(itir.SymRef)
        .to_list()
    )

    extracted = [
        extract_function(stencil, f"{program.id}_stencil_{UIDs.sequential_id()}")
        for stencil in inlined_stencils
    ]

    program = add_fundefs(program, [fundef for _, fundef in extracted])
    program = cast(
        itir.FencilDefinition,
        replace_nodes(
            program, {id(stencil): ref for stencil, (ref, _) in zip(inlined_stencils, extracted)}
        ),
    )
    return program


def generate(program: itir.FencilDefinition, *, grid_type: str, **kwargs: Any) -> str:
    transformed = program
    transformed = apply_common_transforms(
        program,
        use_tmps=kwargs.get("use_tmps", False),
        offset_provider=kwargs.get("offset_provider", None),
        unroll_reduce=True,
    )
    transformed = extract_fundefs_from_closures(transformed)
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


@fencil_formatter
def format_sourcecode(fencil: itir.FencilDefinition, *arg, **kwargs) -> str:
    return generate(fencil, grid_type=_guess_grid_type(**kwargs), **kwargs)
