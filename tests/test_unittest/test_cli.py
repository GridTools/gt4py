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


class NonCliBackend(backend.BaseBackend):
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
    "debug": r"^\s*debug\s*python\s*No",
    "numpy": r"^\s*numpy\s*python\s*No",
    "gtx86": r"^\s*gtx86\s*c\+\+\s*python\s*No",
    "gtmc": r"^\s*gtmc\s*c\+\+\s*python\s*No",
    "gtcuda": r"^\s*gtcuda\s*cuda\s*python\s*No",
}


@pytest.fixture
def list_backends_line_pattern(backend_name):

    yield BACKEND_ROW_PATTERN_BY_NAME[backend_name]


def test_cli_list_backends(clirunner, list_backends_line_pattern):
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


def test_backend_choice(backend_name):
    assert backend_name in cli.BackendChoice.get_backend_names()
    assert not cli.BackendChoice.is_enabled(backend_name)


def test_unenabled_backend_choice(clirunner, nocli_backend):
    result = clirunner.invoke(cli.gtpyc, ["--backend=nocli"])

    assert result.exit_code == 2
    assert re.findall(r".*Backend is not CLI-enabled\..*", result.output)
