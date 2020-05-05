from click.testing import CliRunner
from gt4py import cli
import pytest
import re


@pytest.fixture
def clirunner():
    yield CliRunner()


@pytest.fixture
def simple_stencil(tmp_path):
    module_file = tmp_path / "stencil.py"
    module_file.write_text(
        (
            "## using-dsl: gtscript\n"
            "\n"
            "\n"
            "@mark_stencil()\n"
            "def init_1(input_field: Field[float]):\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        input_field = 1\n"
        )
    )
    yield module_file


@pytest.fixture
def features_stencil(tmp_path):
    module_file = tmp_path / "features_stencil.py"
    module_file.write_text(
        (
            "## using-dsl: gtscript\n"
            "\n"
            "\n"
            "CONSTANT = 5.9\n"
            "\n"
            "\n"
            "@function\n"
            "def some_operation(field_a, field_b, constant=CONSTANT):\n"
            "    return field_a + constant * field_b\n"
            "\n"
            "\n"
            "@mark_stencil(externals={'my_constant': CONSTANT})\n"
            "def some_stencil(field_a: Field[float], field_b: Field[float]):\n"
            "    from __externals__ import my_constant\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        field_a = some_operation(field_a, field_b, constant=my_constant)\n"
            "\n"
            "\n"
            "@mark_stencil(externals={'fill_value': CONSTANT})\n"
            "def fill(field_a: Field[float]):\n"
            "    from __externals__ import fill_value\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        field_a = fill_value\n"
        )
    )
    yield module_file


def test_gtpyc_help(clirunner):
    """Calling the CLI with --help should print the usage message."""
    result = clirunner.invoke(cli.gtpyc, ["--help"])
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_gtpyc_compile_debug(clirunner, simple_stencil, tmp_path):
    """Compile the test stencil using the debug backend and check for compiled source files."""
    output_path = tmp_path / "gtpyc_test_debug"
    result = clirunner.invoke(
        cli.gtpyc, [f"--output-path={output_path}", "--backend=debug", str(simple_stencil)]
    )
    assert result.exit_code == 0
    print(list(output_path.iterdir()))
    py_files = [path.name for path in output_path.iterdir() if path.suffix == ".py"]
    assert "init_1.py" in py_files


def test_gtpyc_compile_gtx86_pyext(clirunner, simple_stencil, tmp_path):
    """Compile the test stencil binary python extension using the gtx86 backend and check the resulting source structure."""
    output_path = tmp_path / "gtpyc_test_gtx86"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "--bindings=python",
            "--compile-bindings",
            "-O",
            "verbose=True",
            str(simple_stencil),
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


def test_gtpyc_nopy_gtx86(clirunner, simple_stencil, tmp_path):
    """Only generate the c++ files."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc, [f"--output-path={output_path}", "--backend=gtx86", str(simple_stencil)],
    )
    assert result.exit_code == 0
    src_files = [path.name for path in output_path.iterdir()]
    assert ["init_1.hpp", "init_1.cpp"] == src_files


def test_gtpyc_withpy_gtx86(clirunner, simple_stencil, tmp_path):
    """Only generate source files but include python bindings."""
    output_path = tmp_path / "gtpyc_test_nopy"
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=gtx86",
            "--bindings=python",
            str(simple_stencil),
        ],
    )
    assert result.exit_code == 0
    src_files = [path.name for path in output_path.iterdir()]
    src_files.sort()
    assert ["bindings.cpp", "init_1.cpp", "init_1.hpp", "init_1.py"] == src_files


def test_sub_and_multi(clirunner, features_stencil, tmp_path):
    """generate two or more stencils, one using a subroutine."""
    output_path = tmp_path / "gtpyc_test_submulti"
    result = clirunner.invoke(
        cli.gtpyc, [f"--output-path={output_path}", "--backend=debug", str(features_stencil),]
    )
    assert result.exit_code == 0, result.output
    src_files = set(path.name for path in output_path.iterdir())
    assert src_files.issuperset({"some_stencil.py", "fill.py"})


def test_externals(clirunner, features_stencil, tmp_path):
    """override externals on commandline."""
    import json
    import traceback

    output_path = tmp_path / "gtpyc_test_externals"
    my_constant = 3.14
    fill_value = 9.0
    externals = json.dumps({"my_constant": my_constant, "fill_value": fill_value})
    result = clirunner.invoke(
        cli.gtpyc,
        [
            f"--output-path={output_path}",
            "--backend=debug",
            f"--externals={externals}",
            str(features_stencil),
        ],
    )
    assert result.exit_code == 0, traceback.print_tb(result.exc_info[2])
    some_stencil_f = output_path / "some_stencil.py"
    fill_f = output_path / "fill.py"
    assert re.findall(
        rf'["\'\s]*my_constant["\'\s]*[:=]["\'\s]*{my_constant}["\'\s]*',
        some_stencil_f.read_text(),
    )
    assert re.findall(
        rf'["\'\s]*fill_value["\'\s]*[:=]["\'\s]*{fill_value}["\'\s]*', fill_f.read_text()
    )
