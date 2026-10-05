# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests of the `GT4PY_DUMP_STAGES` stage dumping."""

from __future__ import annotations

import warnings

import pytest

import gt4py.next as gtx
from gt4py.next import config, gtfn_cpu, typing as gtx_typing
from gt4py.next.instrumentation import stage_dump
from gt4py.next.otf import runners, workflow


try:
    from gt4py.next.program_processors.runners import dace as dace_backends

    BACKENDS = [gtfn_cpu, dace_backends.run_dace_cpu]
except ImportError:
    BACKENDS = [gtfn_cpu]


#: The step names the standard pipelines announce for a DSL program definition,
#: in pipeline order (`Transforms.__call__`, then the `CompilePipeline` fields).
EXPECTED_STAGES = [
    "func_to_past",
    "past_lint",
    "field_view_prog_args_transform",
    "past_to_itir",
    "translation",
    "bindings",
    "compilation",
]

IDim = gtx.Dimension("IDim")


@gtx.field_operator
def dump_op(a: gtx.Field[gtx.Dims[IDim], gtx.float64]) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
    return a + 1.0


@gtx.program
def dump_prog(
    a: gtx.Field[gtx.Dims[IDim], gtx.float64], out: gtx.Field[gtx.Dims[IDim], gtx.float64]
):
    dump_op(a, out=out)


@pytest.fixture
def clean_stage_hook(monkeypatch):
    """Compile in the calling process with no stage subscriber registered.

    The subscriber may already be registered at import time when the test session
    itself runs under `GT4PY_DUMP_STAGES`, so it is explicitly removed and restored.
    """
    monkeypatch.setattr(config, "BUILD_JOBS_MODE", config.BuildJobsMode.SERIAL)
    runners.reset_default_runner()
    was_enabled = stage_dump.SUBSCRIBER_NAME in workflow.stage_hook.registry
    stage_dump.disable()
    yield
    stage_dump.disable()
    if was_enabled:
        stage_dump.enable()
    runners.reset_default_runner()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: getattr(b, "name", str(b)))
def test_dump_stages_end_to_end(
    backend: gtx_typing.Toolchain, tmp_path, monkeypatch, clean_stage_hook
):
    monkeypatch.setattr(config, "DUMP_STAGES", tmp_path)

    stage_dump.enable()
    try:
        # `dump_stage` only warns when it cannot serialize an artifact, so the
        # warnings are inspected: a stage that cannot be dumped must fail here.
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            dump_prog.with_backend(backend).compile(offset_provider={})
    finally:
        stage_dump.disable()

    dump_failures = [str(w.message) for w in recorded if "Could not dump" in str(w.message)]
    assert not dump_failures

    program_dir = tmp_path / "dump_prog"
    assert program_dir.is_dir()

    dumped = sorted(program_dir.iterdir())
    # The index prefixes are unique and reflect the pipeline order.
    assert [path.name.split("_", 1)[1].rsplit(".", 1)[0] for path in dumped] == EXPECTED_STAGES
    for path in dumped:
        assert path.stat().st_size > 0, f"empty stage dump: {path.name}"

    (translation,) = [path for path in dumped if "translation" in path.name]
    assert "dump_prog" in translation.read_text()


def test_no_dump_without_subscriber(tmp_path, monkeypatch, clean_stage_hook):
    monkeypatch.setattr(config, "DUMP_STAGES", tmp_path)

    dump_prog.with_backend(gtfn_cpu).compile(offset_provider={})

    assert list(tmp_path.iterdir()) == []
