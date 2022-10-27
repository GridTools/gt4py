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

from functional.iterator import ir as itir
from functional.program_processors.codegens.gtfn.gtfn_backend import generate
from functional.program_processors.processor_interface import program_formatter


@program_formatter
def format_sourcecode(program: itir.FencilDefinition, *arg: Any, **kwargs: Any) -> str:
    return generate(program, **kwargs)
