# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
import ast
import textwrap

from eve.pattern_matching import ObjectPattern as P
from functional.ffront.ast_passes.rewrite_lineno import RewriteLineNumbers
from functional.ffront.source_utils import SourceDefinition


def square(a):
    return a ** 2


def test_rewrite_lineno():
    source, filename, starting_line = SourceDefinition.from_function(square)
    ast_node = ast.parse(textwrap.dedent(source)).body[0]
    new_ast_node = RewriteLineNumbers.apply(ast_node, starting_line)
    expected_pattern = P(
        ast.FunctionDef,
        args=P(ast.arguments, args=[P(ast.arg, lineno=8)]),
        body=[P(ast.Return, value=P(ast.BinOp, op=P(ast.Pow, lineno=9), lineno=9))],
        lineno=8,
    )
    expected_pattern.match(new_ast_node, raise_exception=True)
