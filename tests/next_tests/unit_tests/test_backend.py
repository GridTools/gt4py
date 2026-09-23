# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-free tests of the `Transforms` / `CompilePipeline` pipelines and the `Toolchain` partial runs."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import gt4py.next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import backend as next_backend, custom_layout_allocators as next_allocators
from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, stages, workflow
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
        backend=next_backend.CompilePipeline(
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
        backend=next_backend.CompilePipeline(
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

    with pytest.raises(NotImplementedError, match="CompilePipeline") as exc_info:
        toolchain.translate(copy_prog.definition_stage, compile_time_args)

    message = str(exc_info.value)
    assert "monolithic" in message  # names the toolchain
    assert "function" in message  # names the offending backend type
    assert frontend_calls == []  # fails fast, before running the frontend


@pytest.fixture
def emitted_stages():
    """Record every `stage_hook` event raised while the fixture is active."""
    emitted: list[tuple[str, Any]] = []

    def stage_callback(name: str, artifact: Any) -> None:
        emitted.append((name, artifact))

    workflow.stage_hook.register(stage_callback)
    try:
        yield emitted
    finally:
        workflow.stage_hook.remove(stage_callback)


def _empty_itir_program() -> itir.Program:
    return itir.Program(id="noop", function_definitions=[], params=[], declarations=[], body=[])


def test_compile_pipeline_runs_steps_in_order_and_emits_hooks(emitted_stages):
    called: list[str] = []

    def translation(inp: Any) -> str:
        called.append("translation")
        return "source"

    def bindings(inp: str) -> str:
        called.append("bindings")
        return "extension"

    def compilation(inp: str) -> str:
        called.append("compilation")
        return "artifact"

    pipeline = next_backend.CompilePipeline(
        translation=translation, bindings=bindings, compilation=compilation
    )

    result = pipeline(_empty_itir_program())

    assert called == ["translation", "bindings", "compilation"]
    assert result == "artifact"
    assert emitted_stages == [
        ("translation", "source"),
        ("bindings", "extension"),
        ("compilation", "artifact"),
    ]


CPU = core_defs.DeviceType.CPU
CUDA = core_defs.DeviceType.CUDA

# Allocates nothing, so a CUDA allocator can be built without `cupy`.
CUDA_ALLOCATOR = next_allocators.InvalidFieldBufferAllocator(
    device_type=CUDA, exception=RuntimeError("Unit tests must not allocate.")
)


def _identity_step(inp: Any) -> Any:
    return inp


@dataclasses.dataclass(frozen=True)
class _DeviceStep:
    """A pass-through step that declares the device it is configured for."""

    device_type: core_defs.DeviceType

    def __call__(self, inp: Any) -> Any:
        return inp


def _cached(step: workflow.Step) -> workflow.CachedStep:
    return workflow.CachedStep.in_memory(step=step, input_fingerprinter=id)


def _pipeline_on(device_type: core_defs.DeviceType) -> next_backend.CompilePipeline:
    return next_backend.CompilePipeline(
        translation=_cached(_DeviceStep(device_type)),
        bindings=_identity_step,
        compilation=_DeviceStep(device_type),
    )


def _toolchain(backend: Any, allocator: Any) -> next_backend.Toolchain:
    return next_backend.Toolchain(
        name="fake", backend=backend, allocator=allocator, frontend=next_backend.DEFAULT_TRANSFORMS
    )


def test_compile_pipeline_rejects_steps_on_different_devices():
    with pytest.raises(ValueError, match="must target the same device") as exc_info:
        next_backend.CompilePipeline(
            translation=_DeviceStep(CPU), bindings=_identity_step, compilation=_DeviceStep(CUDA)
        )

    message = str(exc_info.value)
    assert "'translation' for 'CPU'" in message
    assert "'compilation' for 'CUDA'" in message


def test_compile_pipeline_device_check_looks_through_cached_step():
    with pytest.raises(ValueError, match="'translation' for 'CUDA'"):
        next_backend.CompilePipeline(
            translation=_cached(_cached(_DeviceStep(CUDA))),
            bindings=_identity_step,
            compilation=_DeviceStep(CPU),
        )


def test_compile_pipeline_device_check_covers_replace():
    pipeline = _pipeline_on(CPU)

    # `dataclasses.replace` is the documented customization route: it must not
    # bypass the check a builder would have done.
    with pytest.raises(ValueError, match="must target the same device"):
        dataclasses.replace(pipeline, translation=_DeviceStep(CUDA))


@pytest.mark.parametrize(
    ("translation", "compilation", "expected"),
    [
        (_identity_step, _identity_step, None),
        (_DeviceStep(CUDA), _identity_step, CUDA),
        (_identity_step, _DeviceStep(CPU), CPU),
        (_cached(_DeviceStep(CUDA)), _DeviceStep(CUDA), CUDA),
    ],
)
def test_compile_pipeline_device_type(translation, compilation, expected):
    pipeline = next_backend.CompilePipeline(
        translation=translation, bindings=_identity_step, compilation=compilation
    )

    assert pipeline.device_type is expected


@pytest.mark.parametrize(
    ("backend", "allocator"),
    [
        (_pipeline_on(CPU), CUDA_ALLOCATOR),
        (_pipeline_on(CUDA), next_allocators.StandardCPUFieldBufferAllocator()),
        # A monolithic backend is checked too if it declares a device.
        (_cached(_DeviceStep(CPU)), CUDA_ALLOCATOR),
    ],
)
def test_toolchain_rejects_allocator_on_another_device(backend, allocator):
    with pytest.raises(ValueError, match="allocates buffers on '.*', but its backend targets"):
        _toolchain(backend, allocator)


def test_toolchain_device_check_covers_replace():
    toolchain = _toolchain(_pipeline_on(CPU), next_allocators.StandardCPUFieldBufferAllocator())

    with pytest.raises(ValueError, match="allocates buffers on 'CUDA'"):
        dataclasses.replace(toolchain, allocator=CUDA_ALLOCATOR)
    with pytest.raises(ValueError, match="its backend targets 'CUDA'"):
        dataclasses.replace(toolchain, backend=_pipeline_on(CUDA))


@pytest.mark.parametrize(
    ("backend", "allocator"),
    [
        (_pipeline_on(CUDA), CUDA_ALLOCATOR),
        # Nothing to compare: the backend or the allocator declares no device.
        (_identity_step, CUDA_ALLOCATOR),
        (
            next_backend.CompilePipeline(
                translation=_identity_step, bindings=_identity_step, compilation=_identity_step
            ),
            CUDA_ALLOCATOR,
        ),
        (_pipeline_on(CUDA), None),
    ],
)
def test_toolchain_accepts_agreeing_or_undeclared_devices(backend, allocator):
    toolchain = _toolchain(backend, allocator)

    assert toolchain.backend is backend


def test_transforms_emits_stages_for_dsl_program(compile_time_args, emitted_stages):
    result = next_backend.DEFAULT_TRANSFORMS(
        workflow.ProgramWithArgs(copy_prog.definition_stage, compile_time_args)
    )

    assert [name for name, _ in emitted_stages] == [
        "func_to_past",
        "past_lint",
        "field_view_prog_args_transform",
        "past_to_itir",
    ]
    assert all(isinstance(artifact, workflow.ProgramWithArgs) for _, artifact in emitted_stages)
    assert isinstance(result, workflow.ProgramWithArgs)
    assert isinstance(result.definition, itir.Program)


def test_transforms_emits_stages_for_dsl_field_operator(compile_time_args, emitted_stages):
    result = next_backend.DEFAULT_TRANSFORMS(
        workflow.ProgramWithArgs(copy_op.definition_stage, compile_time_args)
    )

    assert [name for name, _ in emitted_stages] == [
        "func_to_foast",
        "field_view_op_to_prog",
        "past_lint",
        "field_view_prog_args_transform",
        "past_to_itir",
    ]
    assert isinstance(result.definition, itir.Program)


def test_transforms_emits_stages_for_foast_operator(compile_time_args, emitted_stages):
    result = next_backend.DEFAULT_TRANSFORMS(
        workflow.ProgramWithArgs(copy_op.foast_stage, compile_time_args)
    )

    assert [name for name, _ in emitted_stages] == [
        "field_view_op_to_prog",
        "past_lint",
        "field_view_prog_args_transform",
        "past_to_itir",
    ]
    assert isinstance(result.definition, itir.Program)


def test_transforms_emits_stages_for_past_program(compile_time_args, emitted_stages):
    result = next_backend.DEFAULT_TRANSFORMS(
        workflow.ProgramWithArgs(copy_prog.past_stage, compile_time_args)
    )

    assert [name for name, _ in emitted_stages] == [
        "past_lint",
        "field_view_prog_args_transform",
        "past_to_itir",
    ]
    assert isinstance(result.definition, itir.Program)


def test_transforms_itir_program_is_passthrough(compile_time_args, emitted_stages):
    pair = workflow.ProgramWithArgs(_empty_itir_program(), compile_time_args)

    assert next_backend.DEFAULT_TRANSFORMS(pair) is pair
    assert emitted_stages == []


def test_transforms_aotify_and_replace(compile_time_args, emitted_stages):
    transforms = dataclasses.replace(
        next_backend.DEFAULT_TRANSFORMS, aotify_args=lambda jit_args: compile_time_args
    )
    pair = workflow.ProgramWithArgs(_empty_itir_program(), arguments.JITArgs(args=(), kwargs={}))

    result = transforms(pair)

    assert [name for name, _ in emitted_stages] == ["aotify_args"]
    assert result.args is compile_time_args
    assert result.definition is pair.definition


def test_transforms_rejects_unexpected_input(compile_time_args, emitted_stages):
    with pytest.raises(ValueError, match="Unexpected input"):
        next_backend.DEFAULT_TRANSFORMS(workflow.ProgramWithArgs(42, compile_time_args))

    # The type guard runs before `aotify_args`, so nothing is emitted either.
    with pytest.raises(ValueError, match="Unexpected input"):
        next_backend.DEFAULT_TRANSFORMS(
            workflow.ProgramWithArgs(42, arguments.JITArgs(args=(), kwargs={}))
        )

    assert emitted_stages == []


def test_transforms_steps_are_plain_callables():
    """The pipeline fields hold bare callables, not combinator instances."""
    assert callable(next_backend.DEFAULT_TRANSFORMS.past_lint)
    assert isinstance(next_backend.DEFAULT_TRANSFORMS.past_lint, workflow.CachedStep)
    # A data-only step consumes the bare stage, not the `ProgramWithArgs` pair.
    linted = next_backend.DEFAULT_TRANSFORMS.past_lint(copy_prog.past_stage)
    assert isinstance(linted, ffront_stages.PASTProgramDef)


def test_transforms_step_order_fails_loudly():
    """Overriding the removed step-order hook must break, not silently no-op."""

    @dataclasses.dataclass(frozen=True)
    class SkipLinting(next_backend.Transforms):
        def step_order(self, inp):
            order = super().step_order(inp)
            return [step for step in order if step != "past_lint"]

    with pytest.raises(TypeError, match="'Transforms.step_order' was removed"):
        next_backend.DEFAULT_TRANSFORMS.step_order(None)

    with pytest.raises(TypeError, match="'Transforms.step_order' was removed"):
        SkipLinting().step_order(None)
