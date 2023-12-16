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

import gt4py.eve as eve
import gt4py.next.iterator.ir as itir
import gt4py.next.iterator.pretty_parser as pretty_parser
import gt4py.next.iterator.pretty_printer as pretty_printer
import gt4py.next.program_processors.processor_interface as ppi


class _RemoveITIRSymTypes(eve.NodeTranslator):
    def visit_Sym(self, node: itir.Sym) -> itir.Sym:
        return itir.Sym(id=node.id, dtype=None, kind=None)


@ppi.program_formatter
def format_itir_and_check(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> str:
    # remove types from ITIR as they are not supported for the roundtrip
    root = _RemoveITIRSymTypes().visit(program)
    pretty = pretty_printer.pformat(root)
    parsed = pretty_parser.pparse(pretty)
    assert parsed == root
    return pretty
