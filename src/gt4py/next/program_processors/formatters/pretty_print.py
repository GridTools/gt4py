# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next.iterator.ir as itir
import gt4py.next.iterator.pretty_parser as pretty_parser
import gt4py.next.iterator.pretty_printer as pretty_printer
from gt4py.next.program_processors import program_formatter


@program_formatter.program_formatter
def format_itir_and_check(program: itir.Program, *args: Any, **kwargs: Any) -> str:
    pretty = pretty_printer.pformat(program)
    parsed = pretty_parser.pparse(pretty)
    assert parsed == program
    return pretty
