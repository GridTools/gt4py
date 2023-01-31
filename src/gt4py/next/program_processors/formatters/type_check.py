# GT4Py - GridTools Framework
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

from gt4py.next.iterator import ir as itir, type_inference
from gt4py.next.iterator.transforms import apply_common_transforms, global_tmps
from gt4py.next.program_processors.processor_interface import program_formatter


@program_formatter
def check(program: itir.FencilDefinition, *args, **kwargs) -> str:
    type_inference.pprint(type_inference.infer(program))
    transformed = apply_common_transforms(
        program, lift_mode=kwargs.get("lift_mode"), offset_provider=kwargs["offset_provider"]
    )
    if isinstance(transformed, global_tmps.FencilWithTemporaries):
        transformed = transformed.fencil
    return type_inference.pformat(
        type_inference.infer(transformed, offset_provider=kwargs["offset_provider"])
    )
