from gt4py.gtc.gtcpp.gtcpp import GTComputation
import setuptools
from gt4py import gt2_src_manager, config  # TODO must not include gt4py package
from pathlib import Path

from gt4py.gtc.gtcpp import gtcpp
from gt4py.gtc.gtcpp.gtcpp_codegen import GTCppCodegen

if not gt2_src_manager.has_gt_sources() and not gt2_src_manager.install_gt_sources():
    raise RuntimeError("Missing GridTools sources.")


def build_gridtools_test(source: Path):
    ext_module = setuptools.Extension(
        "test",
        [str(source.absolute())],
        include_dirs=[config.GT2_INCLUDE_PATH],
        language="c++",
    )
    args = ["build_ext", "--build-temp=" + source.stem, "--build-lib=" + source.stem, "--force"]
    setuptools.setup(
        name="test",
        ext_modules=[
            ext_module,
        ],
        script_args=args,
    )


# name: SymbolName
#     parameters: List[ParamArg]  # ?
#     temporaries: List[Temporary]
#     multi_stages: List[GTMultiStage]  # TODO at least one


def test_building(tmp_path):

    prog = gtcpp.Program(
        name="test",
        parameters=[],
        functors=[],
        gt_computation=GTComputation(name="test", parameters=[], temporaries=[], multi_stages=[]),
    )

    code = GTCppCodegen.apply(prog)

    tmp_src = tmp_path / "test.cpp"

    tmp_src.write_text(code)

    print("tmp_path: " + str(tmp_src))
    print(code)

    # build_gridtools_test(tmp_src)
