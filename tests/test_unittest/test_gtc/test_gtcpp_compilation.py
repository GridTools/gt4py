# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from pathlib import Path

import pytest
import setuptools

from gt4py import (  # TODO(havogt) this is a dependency from gtc tests to gt4py, ok?
    config,
    gt_src_manager,
)
from gtc.gtcpp.gtcpp import GTApplyMethod, Intent, Program
from gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gtc.gtcpp.oir_to_gtcpp import _extract_accessors

from .gtcpp_utils import (
    ArgFactory,
    FieldDeclFactory,
    GTAccessorFactory,
    GTApplyMethodFactory,
    GTComputationCallFactory,
    GTFunctorFactory,
    GTParamListFactory,
    IfStmtFactory,
    ProgramFactory,
)
from .utils import match


if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
    raise RuntimeError("Missing GridTools sources.")


def build_gridtools_test(tmp_path: Path, code: str):
    tmp_src = tmp_path / "test.cpp"
    tmp_src.write_text(code)

    ext_module = setuptools.Extension(
        "test",
        [str(tmp_src.absolute())],
        include_dirs=[config.GT2_INCLUDE_PATH, config.build_settings["boost_include_path"]],
        language="c++",
    )
    args = [
        "build_ext",
        "--build-temp=" + str(tmp_src.parent),
        "--build-lib=" + str(tmp_src.parent),
        "--force",
    ]
    setuptools.setup(
        name="test",
        ext_modules=[
            ext_module,
        ],
        script_args=args,
    )


def make_compilation_input_and_expected():
    return [
        (ProgramFactory(name="test"), r"auto test"),
        (
            ProgramFactory(functors__0__name="fun"),
            r"struct fun",
        ),
        (
            ProgramFactory(
                functors__0__applies=[],
                functors__0__param_list=GTParamListFactory(
                    accessors=[
                        GTAccessorFactory(
                            id=0,
                            extent__i=(1, 2),
                            extent__j=(-3, -4),
                            extent__k=(10, -10),
                            intent=Intent.INOUT,
                        )
                    ]
                ),
            ),
            r"inout_accessor<0, extent<1,\s*2,\s*-3,\s*-4,\s*10,\s*-10>",
        ),
        (
            ProgramFactory(),
            r"void\s*apply\(",
        ),
        (ProgramFactory(parameters=[FieldDeclFactory(name="my_param")]), r"my_param"),
        (
            ProgramFactory(
                parameters=[FieldDeclFactory(name="outer_param")],
                functors=[GTFunctorFactory(name="fun")],
                gt_computation=GTComputationCallFactory(
                    multi_stages__0__stages__0__functor="fun",
                    multi_stages__0__stages__0__args=[ArgFactory(name="outer_param")],
                    arguments=[ArgFactory(name="outer_param")],
                    temporaries=[],
                ),
            ),
            r"",
        ),
        (
            ProgramFactory(
                functors=[GTFunctorFactory(name="fun", applies__0__body=[])],
                gt_computation=GTComputationCallFactory(multi_stages__0__stages__0__functor="fun"),
            ),
            r"",
        ),
    ]


@pytest.mark.parametrize("gtcpp_program,expected_regex", make_compilation_input_and_expected())
def test_program_compilation_succeeds(tmp_path, gtcpp_program, expected_regex):
    assert isinstance(gtcpp_program, Program)
    code = GTCppCodegen.apply(gtcpp_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, expected_regex)
    build_gridtools_test(tmp_path, code)


def _embed_apply_method_in_program(apply_method: GTApplyMethod):
    accessors = _extract_accessors(apply_method)
    return ProgramFactory(
        functors__0__applies__0=apply_method,
        functors__0__param_list=GTParamListFactory(accessors=accessors),
    )


@pytest.mark.parametrize(
    "apply_method,expected_regex",
    [
        (GTApplyMethodFactory(), r"apply"),
        (
            GTApplyMethodFactory(body__0__left__name="foo", body__0__right__name="bar"),
            r"foo.*=.*bar",
        ),
        (
            GTApplyMethodFactory(body__0=IfStmtFactory()),
            r"if",
        ),
    ],
)
def test_apply_method_compilation_succeeds(tmp_path, apply_method, expected_regex):
    # This test could be improved by just compiling the body
    # and introducing fakes for `eval` and `gridtools::accessor`.
    assert isinstance(apply_method, GTApplyMethod)
    apply_method_code = GTCppCodegen().visit(apply_method, offset_limit=2)
    print(apply_method_code)
    match(apply_method_code, expected_regex)

    build_gridtools_test(
        tmp_path,
        GTCppCodegen.apply(_embed_apply_method_in_program(apply_method), gt_backend_t="cpu_ifirst"),
    )
