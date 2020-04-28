from click.testing import CliRunner
from gt4py import cli
import pytest


@pytest.fixture
def clirunner():
    yield CliRunner()


@pytest.fixture
def test_stencil(tmp_path):
    module_file = tmp_path / "stencil.py"
    module_file.write_text(
        (
            "def init_1(input_field: 'Field[float]'):\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        input_field = 1\n"
        )
    )
    yield module_file


def test_gtpyc_help(clirunner):
    """Calling the CLI with --help should print the usage message."""
    result = clirunner.invoke(cli.gtpyc, ["--help"])
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_gtpyc_compile_debug(clirunner, test_stencil, tmp_path):
    """Compile the test stencil using the debug backend and check for compiled source files."""
    output_path = tmp_path / "gtpyc_test_debug"
    result = clirunner.invoke(
        cli.gtpyc, [f"--output-path={output_path}", "--backend=debug", str(test_stencil)]
    )
    assert result.exit_code == 0
    stencil_path = output_path / "init_1"
    print(list(stencil_path.iterdir()))
    py_files = [path for path in stencil_path.iterdir() if path.suffix == ".py"]
    stencil_modules = [path for path in py_files if path.name.startswith("m_init_1")]
    assert stencil_modules


def test_gtpyc_compile_gtx86(clirunner, test_stencil, tmp_path):
    """Compile the test stencil using the gtx86 backend and check the resulting source structure."""
    output_path = tmp_path / "gtpyc_test_gtx86"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "-O",
            "verbose",
            "True",
            str(test_stencil),
        ],
    )
    assert result.exit_code == 0
    stencil_path = output_path / "init_1"
    print(list(stencil_path.iterdir()))
    py_files = [path for path in stencil_path.iterdir() if path.suffix == ".py"]
    stencil_modules = [path for path in py_files if path.name.startswith("m_init_1")]
    ext_modules = [
        path
        for path in stencil_path.iterdir()
        if path.suffix == ".so" and path.name.startswith("m_init_1")
    ]
    build_paths = [path for path in stencil_path.iterdir() if path.name.endswith("BUILD")]
    assert ext_modules
    assert stencil_modules
    assert build_paths
    build_path = build_paths[0]
    print([path.name for path in build_path.iterdir()])
    cpp_files = [path.name for path in build_path.iterdir() if path.suffix == ".cpp"]
    hpp_files = [path.name for path in build_path.iterdir() if path.suffix == ".hpp"]
    assert "computation.cpp" in cpp_files
    assert "bindings.cpp" in cpp_files
    assert "computation.hpp" in hpp_files


def test_gtpyc_nopy_gtx86(clirunner, test_stencil, tmp_path):
    """Only generate the c++ files."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc,
        [f"--output-path={output_path}", "--backend=gtx86", "--src-only=nopy", str(test_stencil)],
    )
    assert result.exit_code == 0
    src_files = [path.name for path in output_path.iterdir()]
    assert ["init_1.hpp", "init_1.cpp"] == src_files


def test_gtpyc_withpy_gtx86(clirunner, test_stencil, tmp_path):
    """Only generate source files but include python bindings."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "--src-only=withpy",
            str(test_stencil),
        ],
    )
    assert result.exit_code == 0
    src_files = [path.name for path in output_path.iterdir()]
    src_files.sort()
    assert ["bindings.cpp", "init_1.cpp", "init_1.hpp", "init_1.py"] == src_files
