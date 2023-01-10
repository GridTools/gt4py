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
from functional.iterator.transforms.pass_manager import apply_common_transforms
from functional.program_processors.codegens.gtfn.codegen import GTFNCodegen, GTFNIMCodegen
from functional.program_processors.codegens.gtfn.gtfn_ir_to_gtfn_im_ir import GTFN_IM_lowering
from functional.program_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering


def generate(program: itir.FencilDefinition, **kwargs: Any) -> str:
    transformed = program
    offset_provider = kwargs.get("offset_provider")
    do_unroll = not ("imperative" in kwargs and kwargs["imperative"])
    assert isinstance(offset_provider, dict)
    transformed = apply_common_transforms(
        program,
        lift_mode=kwargs.get("lift_mode"),
        offset_provider=offset_provider,
        unroll_reduce=do_unroll,
    )
    gtfn_ir = GTFN_lowering.apply(
        transformed,
        offset_provider=offset_provider,
        column_axis=kwargs.get("column_axis"),
    )
    if "imperative" in kwargs and kwargs["imperative"]:
        gtfn_im_ir = GTFN_IM_lowering().visit(node=gtfn_ir, **kwargs)
        generated_code = GTFNIMCodegen.apply(gtfn_im_ir, **kwargs)
    else:
        generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")
