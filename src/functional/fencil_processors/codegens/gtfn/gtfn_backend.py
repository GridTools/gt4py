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


from typing import Any

import functional.iterator.ir as itir
from eve import codegen
from functional import common
from functional.fencil_processors.codegens.gtfn.codegen import GTFNCodegen
from functional.fencil_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering
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
        transformed,
        grid_type=grid_type,
        offset_provider=offset_provider,
        column_axis=kwargs.get("column_axis"),
    )
    generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def guess_grid_type(**kwargs):
    assert "offset_provider" in kwargs
    return (
        "unstructured"
        if any(isinstance(o, common.Connectivity) for o in kwargs["offset_provider"])
        else "cartesian"
    )
