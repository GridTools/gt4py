# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the `GT4PY_DUMP_STAGES` stage-dump subscriber."""

from __future__ import annotations

import dataclasses
import json

import pytest

from gt4py.next import config
from gt4py.next.instrumentation import stage_dump
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, workflow
from gt4py.next.otf.binding import interface


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DUMP_STAGES", tmp_path)
    yield tmp_path


@pytest.fixture
def serial_jobs(monkeypatch):
    """Silence the `enable` warning about worker processes not dumping."""
    monkeypatch.setattr(config, "BUILD_JOBS_MODE", config.BuildJobsMode.SERIAL)


@pytest.fixture
def restore_subscriber():
    """Restore the subscriber a session started under `GT4PY_DUMP_STAGES` came with."""
    was_enabled = stage_dump.SUBSCRIBER_NAME in workflow.stage_hook.registry
    yield
    stage_dump.disable()
    if was_enabled:
        stage_dump.enable()


def make_program_source(name: str = "prog", source_code: str = "int x;") -> artifacts.ProgramSource:
    return artifacts.ProgramSource(
        entry_point=interface.Function(name, ()),
        source_code=source_code,
        library_deps=(),
        code_spec=artifacts.CPPCodeSpec(),
    )


def test_enable_disable_idempotent(serial_jobs, restore_subscriber):
    stage_dump.disable()

    stage_dump.enable()
    stage_dump.enable()
    assert list(workflow.stage_hook.registry) == [stage_dump.SUBSCRIBER_NAME]
    assert workflow.stage_hook.callbacks == (stage_dump.dump_stage,)

    stage_dump.disable()
    stage_dump.disable()
    assert stage_dump.SUBSCRIBER_NAME not in workflow.stage_hook.registry
    assert workflow.stage_hook.callbacks == ()


def test_enable_warns_when_compiling_in_worker_processes(monkeypatch, restore_subscriber):
    monkeypatch.setattr(config, "BUILD_JOBS_MODE", config.BuildJobsMode.PROCESS)
    stage_dump.disable()

    with pytest.warns(UserWarning, match="only the frontend stages will be dumped"):
        stage_dump.enable()


def test_dump_disabled_writes_nothing(monkeypatch):
    monkeypatch.setattr(config, "DUMP_STAGES", None)
    writes: list[tuple] = []
    monkeypatch.setattr(stage_dump, "_write_unique", lambda *args: writes.append(args))

    stage_dump.dump_stage("translation", make_program_source())

    assert writes == []


def test_dump_program_source(dump_dir):
    stage_dump.dump_stage("translation", make_program_source())

    target = dump_dir / "prog" / "000_translation.cpp"
    assert target.exists()
    assert target.read_text() == "int x;"


def test_dump_index_collision(dump_dir):
    stage_dump.dump_stage("translation", make_program_source(source_code="first"))
    stage_dump.dump_stage("translation", make_program_source(source_code="second"))

    assert (dump_dir / "prog" / "000_translation.cpp").read_text() == "first"
    assert (dump_dir / "prog" / "001_translation.cpp").read_text() == "second"


def test_dump_index_skips_preexisting_file(dump_dir, monkeypatch):
    # A file written by another process (or an earlier run) must never be overwritten,
    # even though this process' index counter still starts at zero.
    monkeypatch.setattr(stage_dump, "_next_index", {})
    (dump_dir / "prog").mkdir(parents=True)
    (dump_dir / "prog" / "000_translation.cpp").write_text("foreign")

    stage_dump.dump_stage("translation", make_program_source(source_code="mine"))

    assert (dump_dir / "prog" / "000_translation.cpp").read_text() == "foreign"
    assert (dump_dir / "prog" / "001_translation.cpp").read_text() == "mine"


def test_dump_index_continues_after_other_processes(dump_dir, monkeypatch):
    # A worker process compiling the second half of the pipeline starts with an
    # empty counter; it must continue the numbering of the frontend dumps the
    # parent already wrote, otherwise a sorted listing loses the pipeline order.
    monkeypatch.setattr(stage_dump, "_next_index", {})
    program_dir = dump_dir / "prog"
    program_dir.mkdir(parents=True)
    for index, stage in enumerate(["func_to_past", "past_lint", "past_to_itir"]):
        (program_dir / f"{index:03d}_{stage}.txt").write_text("earlier")
    (program_dir / "not_indexed.txt").write_text("ignored")

    stage_dump.dump_stage("translation", make_program_source(source_code="mine"))
    stage_dump.dump_stage("bindings", make_program_source(source_code="also mine"))

    assert (program_dir / "003_translation.cpp").read_text() == "mine"
    assert (program_dir / "004_bindings.cpp").read_text() == "also mine"
    assert [
        path.name.split("_", 1)[1].rsplit(".", 1)[0]
        for path in sorted(program_dir.iterdir())
        if path.name[:3].isdigit()
    ] == ["func_to_past", "past_lint", "past_to_itir", "translation", "bindings"]


@pytest.mark.parametrize(
    "code_spec, expected_extension",
    [
        (artifacts.CPPCodeSpec(), "cpp"),  # gtfn: bindings compile with the program
        (artifacts.CUDACodeSpec(), "cu"),
        (artifacts.SDFGCodeSpec(), "py"),  # dace: `bind_sdfg` emits Python
    ],
    ids=["cpp", "cuda", "sdfg"],
)
def test_dump_bindings_extension_follows_binding_language(dump_dir, code_spec, expected_extension):
    program_source = artifacts.ProgramSource(
        entry_point=interface.Function("prog", ()),
        source_code="<program>",
        library_deps=(),
        code_spec=code_spec,
    )
    extension_source = artifacts.ExtensionSource(
        program_source=program_source,
        binding_source=artifacts.BindingSource(source_code="<bindings>", library_deps=()),
    )

    stage_dump.dump_stage("bindings", extension_source)

    target = dump_dir / "prog" / f"000_bindings.{expected_extension}"
    assert target.read_text() == "<bindings>"


def test_dump_bindings_without_binding_source(dump_dir):
    extension_source = artifacts.ExtensionSource(
        program_source=make_program_source(), binding_source=None
    )

    stage_dump.dump_stage("bindings", extension_source)

    assert (dump_dir / "prog" / "000_bindings.txt").read_text() == "(no binding source generated)"


@pytest.mark.parametrize("name", [".", "..", "...", ""])
def test_sanitize_rejects_path_navigation(name):
    assert stage_dump._sanitize(name) == "unknown_program"


def test_dump_opaque_fallback(dump_dir):
    @dataclasses.dataclass
    class Opaque:
        value: int

    artifact = Opaque(value=42)
    stage_dump.dump_stage("mystery", artifact)

    target = dump_dir / "unknown_program" / "000_mystery.txt"
    assert target.read_text() == repr(artifact)


def test_dump_envelope_unwraps(dump_dir):
    program = itir.Program(
        id="my_prog", function_definitions=[], params=[], declarations=[], body=[]
    )
    stage_dump.dump_stage(
        "past_to_itir",
        workflow.ProgramWithArgs(definition=program, args=arguments.CompileTimeArgs.empty()),
    )

    target = dump_dir / "my_prog" / "000_past_to_itir.txt"
    assert target.read_text() == str(program)


def test_dump_sanitizes_program_name(dump_dir):
    stage_dump.dump_stage("translation", make_program_source(name="weird/name"))

    assert (dump_dir / "weird_name" / "000_translation.cpp").exists()


def test_dump_non_text_source_code(dump_dir):
    # The dace backend stores the SDFG as its deserialized JSON object, so
    # `ProgramSource.source_code` is not always a string.
    source = artifacts.ProgramSource(
        entry_point=interface.Function("prog", ()),
        source_code={"type": "SDFG", "nodes": []},
        library_deps=(),
        code_spec=artifacts.SDFGCodeSpec(),
    )

    stage_dump.dump_stage("translation", source)

    target = dump_dir / "prog" / "000_translation.sdfg"
    assert json.loads(target.read_text()) == {"type": "SDFG", "nodes": []}


def test_dump_failure_warns_but_does_not_raise(dump_dir):
    class Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    with pytest.warns(UserWarning, match="Could not dump the artifact of stage 'mystery'"):
        stage_dump.dump_stage("mystery", Unprintable())


def test_dump_write_failure_leaves_no_empty_file(dump_dir, monkeypatch):
    # A failing write must not leave a zero-byte file that looks like an empty stage.
    monkeypatch.setattr(stage_dump, "_source_text", lambda source_code: None)

    with pytest.warns(UserWarning, match="Could not dump"):
        stage_dump.dump_stage("translation", make_program_source())

    assert list((dump_dir / "prog").iterdir()) == []
