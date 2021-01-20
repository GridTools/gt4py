# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the command line interface (CLI)."""

import re
import sys

import pytest
from click.testing import CliRunner

from gt4py import backend, cli
from gt4py.backend.base import CLIBackendMixin


@pytest.fixture
def clirunner():
    """Make the CLiRunner instance conveniently available."""
    yield CliRunner()


@pytest.fixture(
    params=[
        pytest.param(
            name,
            # gtc backends require definition ir as input, for now we skip the tests
            marks=pytest.mark.skipif(
                name.startswith("dawn:") or name.startswith("gtc:"),
                reason="gtc backends not yet supported",
            ),
        )
        for name in list(backend.REGISTRY.keys()) + ["nocli"]
    ],
)
def backend_name(request, nocli_backend):
    """Parametrize by backend name."""
    yield request.param


@pytest.fixture
def clean_imports():
    impdata = [sys.path.copy(), sys.meta_path.copy(), sys.modules.copy()]
    yield
    sys.path, sys.meta_path, sys.modules = impdata


@pytest.fixture
def simple_stencil(tmp_path, clean_imports):
    """Provide a gtscript file with a simple stencil."""
    module_file = tmp_path / "stencil.gt.py"
    module_file.write_text(
        (
            "# [GT] using-dsl: gtscript\n"
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
def nocli_backend(scope="module"):
    """Temporarily register the nocli backend."""
    backend.register(NonCliBackend)
    yield
    backend.remove(NonCliBackend.name)


BACKEND_ROW_PATTERN_BY_NAME = {
    "debug": r"^\s*debug\s*python\s*Yes",
    "numpy": r"^\s*numpy\s*python\s*Yes",
    "gtx86": r"^\s*gtx86\s*c\+\+\s*python\s*Yes",
    "gtmc": r"^\s*gtmc\s*c\+\+\s*python\s*Yes",
    "gtcuda": r"^\s*gtcuda\s*cuda\s*python\s*Yes",
    "gtc:gt:cpu_ifirst": r"^\s*gtc:gt:cpu_ifirst\s*c\+\+\s*python\s*Yes",
    "dawn:gtx86": r"^\s*dawn:gtx86\s*c\+\+\s*python\s*No",
    "dawn:gtmc": r"^\s*dawn:gtmc\s*c\+\+\s*python\s*No",
    "dawn:gtcuda": r"^\s*dawn:gtcuda\s*cuda\s*python\s*No",
    "dawn:naive": r"^\s*dawn:naive\s*c\+\+\s*python\s*No",
    "dawn:cxxopt": r"^\s*dawn:cxxopt\s*c\+\+\s*python\s*No",
    "dawn:cuda": r"^\s*dawn:cuda\s*cuda\s*python\s*No",
    "nocli": r"^\s*nocli\s*\?\s*\?\s*No",
}


@pytest.fixture
def list_backends_line_pattern(backend_name):

    yield BACKEND_ROW_PATTERN_BY_NAME[backend_name]


@pytest.fixture
def backend_enabled(backend_name):
    yield issubclass(backend.from_name(backend_name), CLIBackendMixin)


def test_list_backends(clirunner, list_backends_line_pattern):
    """Test the list-backend subcommand of gtpyc."""
    result = clirunner.invoke(cli.gtpyc, ["list-backends"], catch_exceptions=False)

    assert result.exit_code == 0
    assert re.findall(list_backends_line_pattern, result.output, re.MULTILINE), print(result.output)


def test_gen_silent(clirunner, simple_stencil, tmp_path):
    """Test the --silent flag."""
    result = clirunner.invoke(
        cli.gtpyc,
        [
            "gen",
            "--backend=debug",
            "--output-path",
            str(tmp_path / "test_gen_silent"),
            "--silent",
            str(simple_stencil.absolute()),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert not result.output


def test_gen_missing_arg(clirunner):
    """Test for error if no input path argument is passed to gen."""
    result = clirunner.invoke(cli.gtpyc, ["gen", "--backend=debug"])

    assert result.exit_code == 2
    assert "Missing argument 'INPUT_PATH'." in result.output


def test_backend_choice(backend_name):
    """Test the :py:class:`gt4py.cli.BackendChoice` class interface."""
    assert backend_name in cli.BackendChoice.get_backend_names()


def test_gen_unenabled_backend_choice(clirunner, nocli_backend, simple_stencil, tmp_path):
    """Test the --backend option when an unenabled backend name is passed."""
    result = clirunner.invoke(
        cli.gtpyc,
        [
            "gen",
            "--backend=nocli",
            str(simple_stencil.absolute()),
            "--output-path",
            str(tmp_path / "test_gen_unenabled"),
        ],
    )

    assert result.exit_code == 2
    assert re.findall(r".*Backend is not CLI-enabled\..*", result.output)


def test_gen_enabled_backend_choice(
    clirunner, simple_stencil, backend_name, backend_enabled, tmp_path
):
    """Test if cli runs successfully or fails depending on a backend being enabled."""
    result = clirunner.invoke(
        cli.gtpyc,
        [
            "gen",
            "--backend",
            backend_name,
            str(simple_stencil.absolute()),
            "--output-path",
            str(tmp_path / "test_gen_unenabled"),
        ],
        catch_exceptions=False,
    )

    if backend_enabled:
        assert result.exit_code == 0


def test_gen_gtx86(clirunner, simple_stencil, tmp_path):
    """Only generate the c++ files."""
    output_path = tmp_path / "test_gen_gtx86"
    result = clirunner.invoke(
        cli.gtpyc,
        ["gen", f"--output-path={output_path}", "--backend=gtx86", str(simple_stencil)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    src_path = output_path / "init_1_src"
    src_files = [path.name for path in src_path.iterdir()]
    assert set(["computation.hpp", "computation.cpp"]) == set(src_files), result.output


def test_backend_option_order(clirunner, simple_stencil, tmp_path):
    """Make sure the order in which --backend and --option are passed does not matter."""
    output_path1 = tmp_path / "backend_first"
    # putting the option after the backend should definitely check against the chosen backend
    assert (
        clirunner.invoke(
            cli.gtpyc,
            [
                "gen",
                f"--output-path={output_path1}",
                "--backend=numpy",
                "-O",
                "ignore_np_errstate=True",
                str(simple_stencil),
            ],
            catch_exceptions=False,
        ).exit_code
        == 0
    )
    # putting the option before the backend should still check against the right backend
    assert (
        clirunner.invoke(
            cli.gtpyc,
            [
                "gen",
                f"--output-path={output_path1}",
                "-O",
                "ignore_np_errstate=True",
                "--backend=numpy",
                str(simple_stencil),
            ],
            catch_exceptions=False,
        ).exit_code
        == 0
    )


def test_write_computation_src(tmp_path, simple_stencil):
    builder = cli.GTScriptBuilder(
        simple_stencil, output_path=tmp_path, backend=backend.from_name("debug")
    )
    toplevel = "test_write_computation_src"
    test_src = {
        toplevel: {
            "include": {"header.hpp": "#pragma once"},
            "src": {"main.cpp": "#include <header.hpp>"},
        }
    }
    builder.write_computation_src(tmp_path, test_src)
    top = tmp_path / toplevel
    inc = top / "include"
    header = inc / "header.hpp"
    src = top / "src"
    main = src / "main.cpp"
    assert top.exists() and top.is_dir()
    assert inc.exists() and inc.is_dir()
    assert src.exists() and src.is_dir()
    assert header.exists() and header.read_text() == test_src[toplevel]["include"]["header.hpp"]
    assert main.exists() and main.read_text() == test_src[toplevel]["src"]["main.cpp"]
