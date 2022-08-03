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


from typing import Any, cast

import functional.iterator.ir as itir
from eve import codegen
from eve.utils import UIDs
from functional.fencil_processors.codegens.gtfn.codegen import GTFNCodegen
from functional.fencil_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering
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


def generate(program: itir.FencilDefinition, *, grid_type=None, **kwargs: Any) -> str:
    transformed = program
    offset_provider = kwargs.get("offset_provider", None)
    transformed = apply_common_transforms(
        program,
        use_tmps=kwargs.get("use_tmps", False),
        offset_provider=offset_provider,
        unroll_reduce=True,
    )
    transformed = extract_fundefs_from_closures(transformed)
    gtfn_ir = GTFN_lowering(grid_type=grid_type).visit(transformed, offset_provider=offset_provider)
    generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")
