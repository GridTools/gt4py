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
