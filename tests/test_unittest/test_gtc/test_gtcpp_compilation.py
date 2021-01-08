from pathlib import Path

import pytest
import setuptools

from gt4py import config, gt2_src_manager  # TODO must not include gt4py package or ok for test?
from gt4py.gtc.common import DataType
from gt4py.gtc.gtcpp.gtcpp import GTAccessor, GTApplyMethod, GTExtent, GTStage, Intent, Program
from gt4py.gtc.gtcpp.gtcpp_codegen import GTCppCodegen
from gt4py.gtc.gtcpp.oir_to_gtcpp import _extract_accessors

from .gtcpp_utils import (
    AssignStmtBuilder,
    GTApplyMethodBuilder,
    GTComputationBuilder,
    GTFunctorBuilder,
    IfStmtBuilder,
    ProgramBuilder,
)
from .utils import match


if not gt2_src_manager.has_gt_sources() and not gt2_src_manager.install_gt_sources():
    raise RuntimeError("Missing GridTools sources.")


def build_gridtools_test(tmp_path: Path, code: str):
    tmp_src = tmp_path / "test.cpp"
    tmp_src.write_text(code)

    ext_module = setuptools.Extension(
        "test",
        [str(tmp_src.absolute())],
        include_dirs=[config.GT2_INCLUDE_PATH],
        language="c++",
        # extra_compile_args=["-Wno-unknown-pragmas"],
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
        (ProgramBuilder("test").build(), r"auto test"),
        (
            ProgramBuilder("test").add_functor(GTFunctorBuilder("fun").build()).build(),
            r"struct fun",
        ),
        (
            ProgramBuilder("test")
            .add_functor(
                GTFunctorBuilder("fun")
                .add_accessor(
                    GTAccessor(
                        name="field",
                        id=0,
                        extent=GTExtent(i=(1, 2), j=(-3, -4), k=(10, -10)),
                        intent=Intent.INOUT,
                    )
                )
                .build()
            )
            .build(),
            r"inout_accessor<0, extent<1,\s*2,\s*-3,\s*-4,\s*10,\s*-10>",
        ),
        (
            ProgramBuilder("test")
            .add_functor(
                GTFunctorBuilder("fun").add_apply_method().build(),
            )
            .build(),
            r"void\s*apply\(",
        ),
        (ProgramBuilder("test").add_parameter("my_param", DataType.FLOAT64).build(), r"my_param"),
        (
            ProgramBuilder("test")
            .add_parameter("outer_param", DataType.FLOAT64)
            .add_functor(GTFunctorBuilder("fun").build())
            .gt_computation(
                GTComputationBuilder("test").add_stage(GTStage(functor="fun", args=[])).build()
            )
            .build(),
            r"",
        ),
    ]


@pytest.mark.parametrize("gtcpp_program,expected_regex", make_compilation_input_and_expected())
def test_program_compilation_succeeds(tmp_path, gtcpp_program, expected_regex):
    assert isinstance(gtcpp_program, Program)
    code = GTCppCodegen.apply(gtcpp_program)
    print(code)
    match(code, expected_regex)
    build_gridtools_test(tmp_path, code)


def _embed_apply_method_in_program(apply_method: GTApplyMethod):
    accessors = _extract_accessors(apply_method)
    return (
        ProgramBuilder("test")
        .add_functor(
            GTFunctorBuilder("fun").add_accessors(accessors).add_apply_method(apply_method).build()
        )
        .build()
    )


@pytest.mark.parametrize(
    "apply_method,expected_regex",
    [
        (GTApplyMethodBuilder().build(), r"apply"),
        (
            GTApplyMethodBuilder().add_stmt(AssignStmtBuilder("a", "b").build()).build(),
            r"a.*=.*b",
        ),
        (
            GTApplyMethodBuilder()
            .add_stmt(IfStmtBuilder().true_branch(AssignStmtBuilder().build()).build())
            .build(),
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
        tmp_path, GTCppCodegen.apply(_embed_apply_method_in_program(apply_method))
    )
