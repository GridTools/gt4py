# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
