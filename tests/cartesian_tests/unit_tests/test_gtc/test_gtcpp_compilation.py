# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import gridtools_cpp
import pytest
import setuptools

from gt4py.cartesian.backend import pyext_builder
from gt4py.cartesian.gtc.gtcpp.gtcpp import GTApplyMethod, Intent, Program
from gt4py.cartesian.gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gt4py.cartesian.gtc.gtcpp.oir_to_gtcpp import _extract_accessors

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


def build_gridtools_test(tmp_path: Path, code: str):
    tmp_src = tmp_path / "test.cpp"
    tmp_src.write_text(code)

    extra_compile_args = ["-std=c++17"]
    ext_module = setuptools.Extension(
        "test",
        [str(tmp_src.absolute())],
        include_dirs=[gridtools_cpp.get_include_dir()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    args = [
        "build_ext",
        "--build-temp=" + str(tmp_src.parent),
        "--build-lib=" + str(tmp_src.parent),
        "--force",
    ]
    pyext_builder.setuptools_setup(
        name="test", ext_modules=[ext_module], script_args=args, build_ext_class=None
    )


def make_compilation_input_and_expected():
    return [
        (ProgramFactory(name="test"), r"auto test"),
        (ProgramFactory(functors__0__name="fun"), r"struct fun"),
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
        (ProgramFactory(), r"void\s*apply\("),
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
    accessors = _extract_accessors(apply_method, set())
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
        (GTApplyMethodFactory(body__0=IfStmtFactory()), r"if"),
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
