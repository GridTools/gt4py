from click.testing import CliRunner
from gt4py import cli
import pytest


@pytest.fixture
def clirunner():
    yield CliRunner()


def test_gtpyc_help(clirunner):
    """Calling the CLI with --help should print the usage message."""
    result = clirunner.invoke(cli.gtpyc, ['--help'])
    assert result.exit_code == 0
    assert 'Usage: ' in result.output
