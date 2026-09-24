# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.eve import codegen
from gt4py.next import config
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners import roundtrip


@pytest.fixture
def empty_program():
    return itir.Program(id="empty", function_definitions=[], params=[], declarations=[], body=[])


@pytest.fixture
def formatter_spy(monkeypatch):
    formatted_sources = []

    def spy_formatter(source, **kwargs):
        formatted_sources.append(source)
        return source

    monkeypatch.setattr(codegen, "format_python_source", spy_formatter)
    monkeypatch.setattr(roundtrip, "_SOURCE_CACHE", {})
    return formatted_sources


@pytest.mark.parametrize("flag", [True, False])
def test_format_source_default_captures_config_at_creation(monkeypatch, flag):
    monkeypatch.setattr(config, "FORMAT_SOURCES", flag)
    step = roundtrip.Roundtrip()
    monkeypatch.setattr(config, "FORMAT_SOURCES", not flag)

    assert step.format_source is flag


@pytest.mark.parametrize("format_source", [True, False])
def test_generate_source_formatting(empty_program, formatter_spy, format_source):
    roundtrip._generate_source(
        empty_program,
        debug=False,
        format_source=format_source,
        use_embedded=True,
        offset_provider={},
        transforms=lambda ir, offset_provider: ir,
    )

    assert len(formatter_spy) == int(format_source)


@pytest.mark.parametrize("format_source", [True, False])
def test_roundtrip_step_formatting(empty_program, formatter_spy, format_source):
    step = roundtrip.Roundtrip(
        transforms=lambda ir, offset_provider: ir, format_source=format_source
    )
    step(
        stages.CompilableProgramDef(
            data=empty_program,
            args=arguments.CompileTimeArgs.from_concrete(offset_provider={}),
        )
    )

    assert len(formatter_spy) == int(format_source)
