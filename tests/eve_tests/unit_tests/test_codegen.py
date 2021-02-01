# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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

from typing import Callable, Optional, Set, Type

import pytest

import eve
import eve.codegen

from .test_utils import name_with_cases  # noqa: F401


# -- Name tests --
def test_name(name_with_cases):  # noqa: F811  # pytest fixture not detected
    name = eve.codegen.Name(name_with_cases.pop("words"))
    for case, cased_string in name_with_cases.items():
        assert name.as_case(case) == cased_string
        for other_case, other_cased_string in name_with_cases.items():
            if other_case == eve.utils.CaseStyleConverter.CASE_STYLE.CONCATENATED:
                with pytest.raises(ValueError, match="split a simply concatenated"):
                    other_name = eve.codegen.Name.from_string(other_cased_string, other_case)
                    assert other_name.as_case(case) == cased_string
            else:
                other_name = eve.codegen.Name.from_string(other_cased_string, other_case)
                assert other_name.as_case(case) == cased_string


# -- Template tests --
def fmt_tpl_maker(skeleton, keys, valid=True):
    if valid:
        transformed_keys = {k: "{{{key}}}".format(key=k) for k in keys}
        return eve.codegen.FormatTemplate(skeleton.format(**transformed_keys))
    else:
        return None


def string_tpl_maker(skeleton, keys, valid=True):
    if valid:
        transformed_keys = {k: "${}".format(k) for k in keys}
        return eve.codegen.StringTemplate(skeleton.format(**transformed_keys))
    else:
        return None


def jinja_tpl_maker(skeleton, keys, valid=True):
    if valid:
        transformed_keys = {k: "{{{{ {} }}}}".format(k) for k in keys}
    else:
        transformed_keys = {k: "{{{{ --%{} }}}}".format(k) for k in keys}
    return eve.codegen.JinjaTemplate(skeleton.format(**transformed_keys))


def mako_tpl_maker(skeleton, keys, valid=True):
    if valid:
        transformed_keys = {k: "${{{}}}".format(k) for k in keys}
    else:
        transformed_keys = {k: "${{$<%$${}}}".format(k) for k in keys}
    return eve.codegen.MakoTemplate(skeleton.format(**transformed_keys))


@pytest.fixture(params=[fmt_tpl_maker, string_tpl_maker, jinja_tpl_maker, mako_tpl_maker])
def template_maker(request) -> Callable[[str, Set[str], bool], Optional[eve.codegen.Template]]:
    return request.param


def test_template_definition(template_maker):
    skeleton = "aaa {s} bbbb {i} cccc"
    data = {"s": "STRING", "i": 1}

    template_maker(skeleton, data.keys())
    with pytest.raises(eve.codegen.TemplateDefinitionError):
        t = template_maker(skeleton, data.keys(), False)
        if t is None:
            # Some template engines do not check templates at definition
            raise eve.codegen.TemplateDefinitionError


def test_template_rendering(template_maker):
    skeleton = "aaa {s} bbbb {i} cccc"
    data = {"s": "STRING", "i": 1}
    template = template_maker(skeleton, data.keys())
    assert template.render(**data) == "aaa STRING bbbb 1 cccc"
    assert template.render(data, i=2) == "aaa STRING bbbb 2 cccc"

    with pytest.raises(eve.codegen.TemplateRenderingError):
        template.render()


# -- TemplatedGenerator tests --
class _BaseTestGenerator(eve.codegen.TemplatedGenerator):
    KEYWORDS = ("BASE", "ONE")

    def visit_IntKind(self, node, **kwargs):
        return f"ONE INTKIND({node.value})"

    def visit_SourceLocation(self, node, **kwargs):
        return f"SourceLocation<line:{node.line}, column:{node.column}, source: {node.source}>"

    LocationNode = eve.codegen.FormatTemplate("LocationNode {{{loc}}}")

    SimpleNode = eve.codegen.JinjaTemplate(
        "|{{ bool_value }}, {{ int_value }}, {{ float_value }}, {{ str_value }}, {{ bytes_value }}, "
        "{{ int_kind }}, {{ _this_node.str_kind.__class__.__name__ }}|"
    )

    def visit_SimpleNode(self, node, **kwargs):
        return f"SimpleNode {{{self.generic_visit(node, **kwargs)}}}"

    CompoundNode = eve.codegen.MakoTemplate(
        """
----CompoundNode [BASE]----
    - location: ${location}
    - simple: ${simple}
    - simple_opt: <has_optionals ? (${_this_node.simple_opt.int_value is not None}, ${_this_node.simple_opt.float_value is not None}, ${_this_node.simple_opt.str_value is not None})>
    - other_simple_opt: <is_present ? ${_this_node.other_simple_opt is not None}>
"""
    )

    def visit_CompoundNode(self, node, **kwargs):
        return "TemplatedGenerator result:\n" + self.generic_visit(node, **kwargs)


class _InheritedTestGenerator(_BaseTestGenerator):
    KEYWORDS = ("INHERITED", "OTHER")

    def visit_IntKind(self, node, **kwargs):
        return f"OTHER INTKIND({node.value})"

    CompoundNode = eve.codegen.MakoTemplate(
        """
----CompoundNode [INHERITED]----
    - location: ${location}
    - simple: ${simple}
    - simple_opt: <has_optionals ? (${_this_node.simple_opt.int_value is not None}, ${_this_node.simple_opt.float_value is not None}, ${_this_node.simple_opt.str_value is not None})>
    - other_simple_opt: <is_present ? ${_this_node.other_simple_opt is not None}>
"""
    )


class _FaultyTestGenerator1(eve.codegen.TemplatedGenerator):
    CompoundNode = eve.codegen.FormatTemplate(
        """
Line
Another {MISSING_loc}"""
    )


class _FaultyTestGenerator2(eve.codegen.TemplatedGenerator):
    CompoundNode = eve.codegen.StringTemplate(
        """
Line
Another $f{ffloc"""
    )


class _FaultyTestGenerator3(eve.codegen.TemplatedGenerator):
    CompoundNode = eve.codegen.JinjaTemplate(
        """
|{{ bool_value }}, {{ MISSING_int_value }}, {{ float_value }}, {{ str_value }}, {{ bytes_value }},
        {{ int_kind }}, {{ _this_node.str_kind.__class__.WRONG__name__ }}|
    """
    )


class _FaultyTestGenerator4(eve.codegen.TemplatedGenerator):
    CompoundNode = eve.codegen.MakoTemplate(
        """
----CompoundNode [BASE]----
    - location: ${location}
    - simple: ${MISSING_simple}
    - simple_opt: <has_optionals ? (${_this_node.simple_opt.int_value is not None}, ${_this_node.simple_opt.float_value is not None}, ${_this_node.simple_opt.str_value is not None})>
    - other_simple_opt: <is_present ? ${_this_node.other_simple_opt is not None}>
"""
    )


@pytest.fixture(params=[_BaseTestGenerator, _InheritedTestGenerator])
def templated_generator(request) -> Type[eve.codegen.TemplatedGenerator]:
    return request.param


@pytest.fixture(
    params=[
        _FaultyTestGenerator1,
        _FaultyTestGenerator2,
        _FaultyTestGenerator3,
        _FaultyTestGenerator4,
    ]
)
def faulty_templated_generator(request) -> Type[eve.codegen.TemplatedGenerator]:
    return request.param


def test_templated_generator(templated_generator, fixed_compound_node):
    rendered_code = templated_generator.apply(fixed_compound_node)
    assert rendered_code.find("TemplatedGenerator result:\n") >= 0
    assert rendered_code.find("----CompoundNode [") >= 0
    assert rendered_code.find("LocationNode {") >= 0
    assert rendered_code.find("SimpleNode {|") >= 0
    assert rendered_code.find("<has_optionals ? (True, True, False)>") >= 0
    assert rendered_code.find("<is_present ? False>") >= 0

    for keyword in templated_generator.KEYWORDS:
        assert rendered_code.find(keyword) >= 0


def test_templated_generator_exceptions(faulty_templated_generator, fixed_compound_node):
    with pytest.raises(eve.codegen.TemplateRenderingError, match="when rendering node"):
        faulty_templated_generator.apply(fixed_compound_node)
