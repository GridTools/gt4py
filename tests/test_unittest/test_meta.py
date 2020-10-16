import ast

import gt4py.utils as gt_util


class TestFullNameCreator:
    def test_name_only(self):
        name = "simple_name"
        expr = ast.parse(name, mode="eval")
        assert gt_util.meta.get_full_name(expr) == name

    def test_nested_attribute(self):
        name = "module.submodule.name"
        expr = ast.parse(name, mode="eval")
        assert gt_util.meta.get_full_name(expr) == name
