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


from functional.iterator import type_inference
from functional.iterator.processor_interface import fencil_formatter
from functional.iterator.transforms import apply_common_transforms


@fencil_formatter
def check(root, *args, **kwargs) -> str:
    type_inference.pprint(type_inference.infer(root))
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps"), offset_provider=kwargs["offset_provider"]
    )
    return type_inference.pformat(type_inference.infer(transformed))
