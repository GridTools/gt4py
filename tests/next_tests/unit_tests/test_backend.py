# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-free tests of the `Toolchain` partial runs."""

from __future__ import annotations

from typing import Any

import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, custom_layout_allocators as next_allocators
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, recipes, stages, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_specifications as ts


IDim = gtx.Dimension("IDim")


@gtx.field_operator
def copy_op(a: gtx.Field[gtx.Dims[IDim], gtx.float64]) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
    return a


@gtx.program
def copy_prog(
    a: gtx.Field[gtx.Dims[IDim], gtx.float64], out: gtx.Field[gtx.Dims[IDim], gtx.float64]
):
    copy_op(a, out=out)


SENTINEL_SOURCE = artifacts.ProgramSource(
    entry_point=interface.Function("copy_prog", ()),
    source_code="// sentinel",
    library_deps=(),
    code_spec=artifacts.CPPCodeSpec(),
)


@pytest.fixture
def compile_time_args() -> arguments.CompileTimeArgs:
    field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    return arguments.CompileTimeArgs(
        args=(field_type, field_type),
        kwargs={},
        offset_provider={},
        column_axis=None,
        argument_descriptor_contexts={},
    )


def _unreachable_step(inp: Any) -> Any:
    raise AssertionError("This step must not run in a partial toolchain run.")


def test_translate_returns_program_source(compile_time_args):
    seen: list[stages.CompilableProgram] = []

    def fake_translation(inp: stages.CompilableProgram) -> artifacts.ProgramSource:
        seen.append(inp)
        return SENTINEL_SOURCE

    toolchain = next_backend.Toolchain(
        name="fake",
        backend=recipes.OTFCompileWorkflow(
            translation=fake_translation,
            bindings=_unreachable_step,
            compilation=_unreachable_step,
        ),
        allocator=next_allocators.StandardCPUFieldBufferAllocator(),
        frontend=next_backend.DEFAULT_TRANSFORMS,
    )

    result = toolchain.translate(copy_prog.definition_stage, compile_time_args)

    assert result is SENTINEL_SOURCE
    assert len(seen) == 1
    assert isinstance(seen[0], workflow.ProgramWithArgs)
    assert isinstance(seen[0].definition, itir.Program)
    assert seen[0].args == compile_time_args


def test_translate_emits_stage_hook(compile_time_args):
    emitted: list[tuple[str, Any]] = []

    def stage_callback(name: str, artifact: Any) -> None:
        emitted.append((name, artifact))

    toolchain = next_backend.Toolchain(
        name="fake",
        backend=recipes.OTFCompileWorkflow(
            translation=lambda inp: SENTINEL_SOURCE,
            bindings=_unreachable_step,
            compilation=_unreachable_step,
        ),
        allocator=next_allocators.StandardCPUFieldBufferAllocator(),
        frontend=next_backend.DEFAULT_TRANSFORMS,
    )

    workflow.stage_hook.register(stage_callback)
    try:
        toolchain.translate(copy_prog.definition_stage, compile_time_args)
    finally:
        workflow.stage_hook.remove(stage_callback)

    translation_events = [(name, artifact) for name, artifact in emitted if name == "translation"]
    assert translation_events == [("translation", SENTINEL_SOURCE)]
    # The frontend steps are announced too, and the translation comes last.
    assert emitted[-1][0] == "translation"
    assert "past_to_itir" in [name for name, _ in emitted]


def test_translate_rejects_monolithic_backend(compile_time_args):
    frontend_calls: list[Any] = []

    def recording_frontend(inp: Any) -> Any:
        frontend_calls.append(inp)
        raise AssertionError("The frontend must not run for an unsupported compile pipeline.")

    toolchain = next_backend.Toolchain(
        name="monolithic",
        backend=_unreachable_step,
        allocator=next_allocators.StandardCPUFieldBufferAllocator(),
        frontend=recording_frontend,
    )

    with pytest.raises(NotImplementedError, match="OTFCompileWorkflow") as exc_info:
        toolchain.translate(copy_prog.definition_stage, compile_time_args)

    message = str(exc_info.value)
    assert "monolithic" in message  # names the toolchain
    assert "function" in message  # names the offending backend type
    assert frontend_calls == []  # fails fast, before running the frontend
