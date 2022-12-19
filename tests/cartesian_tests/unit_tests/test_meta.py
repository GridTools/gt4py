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

import ast

import gt4py.cartesian.utils as gt_util
from gt4py.cartesian.frontend.gtscript_frontend import PYTHON_AST_VERSION


class TestGetQualifiedName:
    def test_name_only(self):
        name = "simple_name"
        expr = ast.parse(name, mode="eval", feature_version=PYTHON_AST_VERSION).body
        assert gt_util.meta.get_qualified_name_from_node(expr) == name

    def test_nested_attribute(self):
        name = "module.submodule.name"
        expr = ast.parse(name, mode="eval", feature_version=PYTHON_AST_VERSION).body
        assert gt_util.meta.get_qualified_name_from_node(expr) == name
        assert gt_util.meta.get_qualified_name_from_node(expr, as_list=True) == [
            "module",
            "submodule",
            "name",
        ]
