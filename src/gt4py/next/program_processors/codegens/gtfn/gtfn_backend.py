# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import gt4py.next.iterator.ir as itir
from gt4py.eve import codegen
from gt4py.eve.exceptions import EveValueError
from gt4py.next.iterator.transforms.pass_manager import apply_common_transforms
from gt4py.next.program_processors.codegens.gtfn.codegen import GTFNCodegen, GTFNIMCodegen
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_to_gtfn_im_ir import GTFN_IM_lowering
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import GTFN_lowering


def generate(
    program: itir.FencilDefinition, enable_itir_transforms: bool = True, **kwargs: Any
) -> str:
    offset_provider = kwargs.get("offset_provider")
    assert isinstance(offset_provider, dict)

    gtfn_ir = GTFN_lowering.apply(
        program,
        offset_provider=offset_provider,
        column_axis=kwargs.get("column_axis"),
    )

    if kwargs.get("imperative", False):
        gtfn_im_ir = GTFN_IM_lowering().visit(node=gtfn_ir, **kwargs)
        generated_code = GTFNIMCodegen.apply(gtfn_im_ir, **kwargs)
    else:
        generated_code = GTFNCodegen.apply(gtfn_ir, **kwargs)
    return codegen.format_source("cpp", generated_code, style="LLVM")
