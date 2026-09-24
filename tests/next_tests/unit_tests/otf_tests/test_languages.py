# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import config, fingerprinting
from gt4py.next.otf import artifacts
from gt4py.next.otf.binding import interface


def test_header_files_settings_with_cpp_accepted():
    artifacts.ProgramSource(
        entry_point=interface.Function(name="basic_settings_with_cpp", parameters=[]),
        source_code="",
        library_deps=(),
        code_spec=artifacts.CPPCodeSpec(),
    )


@pytest.mark.parametrize("flag", [True, False])
def test_format_source_default_captures_config_at_creation(monkeypatch, flag):
    monkeypatch.setattr(config, "FORMAT_SOURCES", flag)
    code_spec = artifacts.CPPCodeSpec()
    assert code_spec.format_source is flag

    # changing the global option afterwards does not affect existing specs
    monkeypatch.setattr(config, "FORMAT_SOURCES", not flag)
    assert code_spec.format_source is flag
    assert artifacts.CPPCodeSpec().format_source is not flag


@pytest.mark.parametrize("flag", [True, False])
def test_format_source_is_part_of_fingerprint(monkeypatch, flag):
    monkeypatch.setattr(config, "FORMAT_SOURCES", flag)
    assert fingerprinting.strict_fingerprinter(
        artifacts.CPPCodeSpec()
    ) != fingerprinting.strict_fingerprinter(artifacts.CPPCodeSpec(format_source=not flag))


@pytest.mark.parametrize("flag", [True, False])
def test_sdfg_code_spec_ignores_format_sources_config(monkeypatch, flag):
    monkeypatch.setattr(config, "FORMAT_SOURCES", flag)
    assert artifacts.SDFGCodeSpec().format_source is False


def test_format_source_follows_code_spec():
    source = "x=( 1 )"

    assert artifacts.format_source(artifacts.PythonCodeSpec(format_source=False), source) == source
    assert (
        artifacts.format_source(artifacts.PythonCodeSpec(format_source=True), source) == "x = 1\n"
    )
