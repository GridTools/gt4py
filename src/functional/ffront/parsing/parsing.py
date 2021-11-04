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
import inspect
import textwrap
from typing import Callable


class EmptyReturnException(Exception):
    ...


class FieldOperatorSyntaxError(SyntaxError):
    def __init__(self, msg="", *, lineno=0, offset=0, filename=None):
        self.msg = "Invalid Field Operator Syntax: " + msg
        self.lineno = lineno
        self.offset = offset
        self.filename = filename


def get_ast_from_func(func: Callable) -> ast.stmt:
    if inspect.getabsfile(func) == "<string>":
        raise ValueError(
            "Can not create field operator from a function that is not in a source file!"
        )
    source = textwrap.dedent(inspect.getsource(func))
    return ast.parse(source).body[0]
