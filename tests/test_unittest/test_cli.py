"""Unit tests for the command line interface (CLI)."""
import re

import pytest
from click.testing import CliRunner

from gt4py import cli
from gt4py import backend


@pytest.fixture
def clirunner():
    """Make the CLiRunner instance conveniently available."""
    yield CliRunner()


@pytest.fixture(params=backend.REGISTRY.keys())
def backend_name(request):
    """Parametrize by backend name."""
    yield request.param


@pytest.fixture
def simple_stencil(tmp_path):
    """Provide a gtscript file with a simple stencil."""
    module_file = tmp_path / "stencil.gtpy"
    module_file.write_text(
        (
            "## using-dsl: gtscript\n"
            "\n"
            "\n"
            "@lazy_stencil()\n"
            "def init_1(input_field: Field[float]):\n"
            "    with computation(PARALLEL), interval(...):\n"
            "        input_field = 1\n"
        )
    )
    yield module_file


class NonCliBackend(backend.Backend):
    """This represents a non-CLI-enabled backend."""

    name = "nocli"

    def generate(*args, **kwargs):
        pass

    def load(*args, **kwargs):
        pass


@pytest.fixture
def nocli_backend():
    """Temporarily register the nocli backend."""
    backend.register(NonCliBackend)
    yield
    backend.REGISTRY.pop("nocli")


BACKEND_ROW_PATTERN_BY_NAME = {
    "debug": r"^\s*debug\s*python\s*Yes",
    "numpy": r"^\s*numpy\s*python\s*Yes",
    "gtx86": r"^\s*gtx86\s*c\+\+\s*python\s*Yes",
    "gtmc": r"^\s*gtmc\s*c\+\+\s*python\s*Yes",
    "gtcuda": r"^\s*gtcuda\s*cuda\s*python\s*Yes",
    "dawn:gtx86": r"^\s*dawn:gtx86\s*c\+\+\s*python\s*No",
    "dawn:gtmc": r"^\s*dawn:gtmc\s*c\+\+\s*python\s*No",
    "dawn:gtcuda": r"^\s*dawn:gtcuda\s*cuda\s*python\s*No",
    "dawn:naive": r"^\s*dawn:naive\s*c\+\+\s*python\s*No",
    "dawn:cxxopt": r"^\s*dawn:cxxopt\s*c\+\+\s*python\s*No",
    "dawn:cuda": r"^\s*dawn:cuda\s*cuda\s*python\s*No",
}


@pytest.fixture
def list_backends_line_pattern(backend_name):

    yield BACKEND_ROW_PATTERN_BY_NAME[backend_name]


@pytest.fixture
def backend_enabled(backend_name):
    yield not backend_name.startswith("dawn")


@pytest.fixture(params=[[], ["--list-backends"]])
def options_for_silent(request, simple_stencil):
    """These options combinations are tested to not print output when combined with --silent."""
    yield request.param + [str(simple_stencil.absolute())]


def test_list_backends(clirunner, list_backends_line_pattern):
    """
    Test the --list-backend flag of gtpyc.

    Assumptions:

        * cli.gtpyc is available (cli is importable)
        * cli.gtpyc takes a --list-backends option
        * The expected output line has been made available to the
          :py:func`list_backends_line_pattern` fixture

    Actions:

        .. code-block:: bash

            $ gtpyc --list-backends

    Outcome:

        gtpyc echos a table with a line per backend, listing the
        primary and secondary language and CLI compatibility for each.

    """
    result = clirunner.invoke(cli.gtpyc, ["--list-backends"], catch_exceptions=False)

    assert result.exit_code == 0
    assert re.findall(list_backends_line_pattern, result.output, re.MULTILINE)


def test_silent(clirunner, options_for_silent):
    """
    Test the --silent flag.

    Assumptions:

        * The :py:func:`options_for_silent` fixture contains option / arg combinations that should
          be silent.

    Actions:

        .. code-block:: bash

            $ gtpyc --silent [<parametrized option1>, ...] <simple_stencil_path>

    Outcome:

        The stdout output is empty.
    """
    result = clirunner.invoke(cli.gtpyc, ["--silent"] + options_for_silent)

    assert result.exit_code == 0
    assert not result.output


def test_missing_arg(clirunner):
    """
    Test when no input path argument is passed (and also not --list-backends).

    Actions:

        .. code-block:: bash

            $ gtpyc --backend=debug

    Outcome:

        An error message warns the user that no input path was given. The command aborts with error
        code 2.

    """
    result = clirunner.invoke(cli.gtpyc, [])

    assert result.exit_code == 2
    assert "Missing argument '[INPUT_PATH]'." in result.output


def test_backend_choice(backend_name):
    """
    Test the :py:cls:`gt4py.cli.BackendChoice` class interface.
    """
    assert backend_name in cli.BackendChoice.get_backend_names()


def test_unenabled_backend_choice(clirunner, nocli_backend, simple_stencil):
    """
    Test the --backend option when an unenabled backend name is passed.

    The :py:func:`nocli_backend` fixture temporarily injects a dummy backend named "nocli", which
    is not CLI enabled (and provides no functionality in fact).

    Actions:

        .. code-block:: bash

            $ gtpyc --backend=nocli <simple_stencil_path>

    Outcome:

        An error message warns the user that the chosen backend is not enabled for CLI.
        The command aborts with error code 2.

    """
    result = clirunner.invoke(cli.gtpyc, ["--backend=nocli", str(simple_stencil.absolute())])

    assert result.exit_code == 2
    assert re.findall(r".*Backend is not CLI-enabled\..*", result.output)


def test_enabled_backend_choice(clirunner, simple_stencil, backend_name, backend_enabled):
    """
    Test an enabled backend.

    Actions:

        .. code-block:: bash

            $ gtpyc --backend <parametrized backend_name> <simple_stencil path>

    Outcome:

        The command writes the computation source to the current directory.
    """
    result = clirunner.invoke(
        cli.gtpyc, ["--backend", backend_name, str(simple_stencil.absolute())]
    )

    assert result.exit_code == 0 if backend_enabled else 2
