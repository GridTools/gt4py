#!/usr/bin/env python

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

#
import ast
import textwrap

from gt4py.next.ffront.ast_passes.simple_assign import SingleAssignTargetPass


def getlines(node: ast.AST) -> list[str]:
    return ast.unparse(node).splitlines()


def separate_targets(code: str) -> ast.AST:
    return SingleAssignTargetPass.apply(ast.parse(textwrap.dedent(code)))


def test_separate():
    """Multi target assign should be turned into sequence of assigns."""
    lines = getlines(
        separate_targets(
            """
            a = b = c = foo
            """
        )
    )
    assert len(lines) == 4
    assert lines[0] == "__sat_tmp0 = foo"
    assert lines[1] == "a = __sat_tmp0"
    assert lines[2] == "b = __sat_tmp0"
    assert lines[3] == "c = __sat_tmp0"


def test_separate_unpacking():
    """Multi target assign containing unpacking targets does nothing unexpected."""
    lines = getlines(
        separate_targets(
            """
            a = [b, c] = [d, e]
            """
        )
    )
    assert len(lines) == 3
    assert lines[0] == "__sat_tmp0 = [d, e]"
    assert lines[1] == "a = __sat_tmp0"
    assert lines[2] == "[b, c] = __sat_tmp0"
