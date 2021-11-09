#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

from functional.ffront.ast_passes.simple_assign import SingleAssignTargetPass, UnpackedAssignPass


def getlines(node: ast.AST) -> list[str]:
    return ast.unparse(node).splitlines()


def separate_targets(code: str) -> ast.AST:
    return SingleAssignTargetPass.apply(ast.parse(textwrap.dedent(code)))


def unpack_targets(code: str) -> ast.AST:
    return UnpackedAssignPass.apply(ast.parse(textwrap.dedent(code)))


def test_separate():
    """Multi target assign should be turned into sequence of assigns."""
    lines = getlines(
        separate_targets(
            """
            a = b = c = foo
            """
        )
    )
    assert len(lines) == 3
    assert lines[0] == "a = foo"
    assert lines[1] == "b = foo"
    assert lines[2] == "c = foo"


def test_separate_unpacking():
    """Multi target assign containing unpacking targets does nothing unexpected."""
    lines = getlines(
        separate_targets(
            """
            a = [b, c] = [d, e]
            """
        )
    )
    assert len(lines) == 2
    assert lines[0] == "a = [d, e]"
    assert lines[1] == "[b, c] = [d, e]"


def test_unpack():
    """Single level unpacking assign is explicitly unpacked."""
    lines = getlines(
        unpack_targets(
            """
            a, b = c, d
            """
        )
    )
    assert len(lines) == 2
    assert lines[0] == "a = (c, d)[0]"
    assert lines[1] == "b = (c, d)[1]"


def test_nested_unpack():
    """Nested unpacking accesses deeper levels from the rhs expression."""
    lines = getlines(
        unpack_targets(
            """
            a, [b, [c]] = foo()
            """
        )
    )
    assert len(lines) == 3
    assert lines[0] == "a = foo()[0]"
    assert lines[1] == "b = foo()[1][0]"
    assert lines[2] == "c = foo()[1][1][0]"


def test_nested_multi_target_unpack():
    """Unpacking after separating completely linearizes."""
    lines = getlines(
        UnpackedAssignPass.apply(
            separate_targets(
                """
                a = [b, c] = [d, e]
                """
            )
        )
    )
    assert len(lines) == 3
    assert lines[0] == "a = [d, e]"
    assert lines[1] == "b = [d, e][0]"
    assert lines[2] == "c = [d, e][1]"
