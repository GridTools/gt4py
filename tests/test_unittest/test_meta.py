import ast

import gt4py.utils as gt_util


class TestGetQualifiedName:
    def test_name_only(self):
        name = "simple_name"
        expr = ast.parse(name, mode="eval")
        assert gt_util.meta.get_qualified_name(expr) == name

    def test_nested_attribute(self):
        name = "module.submodule.name"
        expr = ast.parse(name, mode="eval")
        assert gt_util.meta.get_qualified_name(expr) == name
