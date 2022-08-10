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

from functional.fencil_processors.processor_interface import fencil_formatter
from functional.iterator import ir
from functional.iterator.pretty_parser import pparse
from functional.iterator.pretty_printer import pformat


@fencil_formatter
def pretty_format_and_check(root: ir.Node, *args: Any, **kwargs: Any) -> str:
    pretty = pformat(root)
    parsed = pparse(pretty)
    assert parsed == root
    return pretty
