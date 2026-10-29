# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend kinds and the internal toolchain in `gt4py.next.backend`."""

import copy
import dataclasses
import importlib.util
from typing import Any

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, custom_layout_allocators as next_allocators
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, recipes, stages, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners import embedded as embedded_runners, gtfn
from gt4py.next.type_system import type_specifications as ts


IDim = gtx.Dimension("IDim")


@dataclasses.dataclass(frozen=True)
class _OtherBackend(next_backend.Backend):
    """A backend that is neither compiled nor embedded."""


def _make_compiled_backend(**kwargs) -> next_backend.CompiledBackend:
    return next_backend.CompiledBackend(
        name=kwargs.pop("name", "test_backend"),
        allocator=kwargs.pop("allocator", next_allocators.StandardCPUFieldBufferAllocator()),
        compiler=kwargs.pop("compiler", lambda compilable: compilable),
        **kwargs,
    )


def test_backend_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        next_backend.Backend(name="backend", allocator=None)


def test_backend_subclass_can_be_instantiated():
    backend = _OtherBackend(name="other", allocator=None)

    assert backend.name == "other"


def test_compiled_backend_defaults():
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    backend = _make_compiled_backend(allocator=allocator)

    assert isinstance(backend, next_backend.Backend)
    assert backend.loader is artifacts.load_artifact
    assert backend.__gt_allocator__ is allocator
    assert [field.name for field in dataclasses.fields(backend)] == [
        "name",
        "allocator",
        "compiler",
        "loader",
    ]


def test_compiled_backend_replace_keeps_other_fields():
    def other_compiler(compilable):
        return compilable

    backend = _make_compiled_backend()
    variant = dataclasses.replace(backend, compiler=other_compiler)

    assert variant.compiler is other_compiler
    assert variant.name == backend.name
    assert variant.allocator is backend.allocator
    assert variant.loader is backend.loader


def test_toolchain_compile_runs_frontend_compiler_and_loader_in_order():
    calls = []
    compile_time_args = arguments.CompileTimeArgs.empty()

    def frontend(program_with_args):
        calls.append(("frontend", program_with_args.definition, program_with_args.args))
        return "compilable"

    def compiler(compilable):
        calls.append(("compiler", compilable))
        return "artifact"

    def loader(artifact):
        calls.append(("loader", artifact))
        return "executable"

    backend = _make_compiled_backend(compiler=compiler, loader=loader)
    result = next_backend.Toolchain(backend, frontend=frontend).compile(
        "definition", compile_time_args
    )

    assert result == "executable"
    assert calls == [
        ("frontend", "definition", compile_time_args),
        ("compiler", "compilable"),
        ("loader", "artifact"),
    ]


def test_toolchain_default_frontend_is_the_global_one():
    assert next_backend.Toolchain(_make_compiled_backend()).frontend is (
        next_backend.DEFAULT_TRANSFORMS
    )


def test_toolchain_is_not_public():
    assert "Toolchain" not in gtx.__all__
    assert "Toolchain" not in gtx.typing.__all__
    assert "Backend" in gtx.typing.__all__


def test_backend_allocator_unwraps_a_factory():
    inner = next_allocators.StandardCPUFieldBufferAllocator()

    class Factory:
        __gt_allocator__ = inner

    assert _OtherBackend(name="other", allocator=Factory()).__gt_allocator__ is inner
    assert _OtherBackend(name="other", allocator=inner).__gt_allocator__ is inner
    assert _OtherBackend(name="other", allocator=np).__gt_allocator__ is np


def test_embedded_backend_defaults_to_numpy():
    backend = next_backend.EmbeddedBackend(name="my_embedded")

    assert isinstance(backend, next_backend.Backend)
    assert not isinstance(backend, next_backend.CompiledBackend)
    assert backend.allocator is np
    assert backend.__gt_allocator__ is np
    assert [field.name for field in dataclasses.fields(backend)] == ["name", "allocator"]


def test_embedded_backend_rejects_a_non_namespace_allocator():
    with pytest.raises(TypeError, match="must be an array namespace"):
        next_backend.EmbeddedBackend(
            name="bad", allocator=next_allocators.StandardCPUFieldBufferAllocator()
        )


def test_embedded_backend_instances():
    assert gtx.embedded_numpy is embedded_runners.embedded_numpy
    assert isinstance(gtx.embedded_numpy, next_backend.EmbeddedBackend)
    assert gtx.embedded_numpy.name == "embedded_numpy"
    assert gtx.embedded_numpy.allocator is np
    assert "embedded_numpy" in gtx.__all__


@pytest.mark.parametrize("name, package", [("embedded_cupy", "cupy"), ("embedded_jax", "jax")])
def test_optional_embedded_backend(name, package):
    if importlib.util.find_spec(package) is None:
        with pytest.raises(AttributeError, match=f"requires the optional '{package}' package"):
            getattr(embedded_runners, name)
    else:
        backend = getattr(embedded_runners, name)
        assert isinstance(backend, next_backend.EmbeddedBackend)
        assert backend.name == name
        assert getattr(embedded_runners, name) is backend


def test_embedded_runners_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'embedded_foo'"):
        embedded_runners.embedded_foo  # noqa: B018 [useless-expression]


@pytest.mark.parametrize(
    "backend",
    [gtx.embedded_numpy, gtfn.run_gtfn, _make_compiled_backend()],
    ids=lambda backend: backend.name,
)
def test_deepcopy_of_a_backend_is_the_backend(backend):
    assert copy.deepcopy(backend) is backend


def _make_copy_program(backend):
    @gtx.field_operator(backend=backend)
    def identity(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
        return a

    @gtx.program(backend=backend)
    def copy_program(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
        identity(a, out=out)

    return copy_program


@pytest.mark.parametrize(
    "backend", [gtx.embedded_numpy, gtfn.run_gtfn], ids=lambda backend: backend.name
)
def test_deepcopy_of_a_program_shares_the_backend(backend):
    program = _make_copy_program(backend)

    program_copy = copy.deepcopy(program)

    assert program_copy is not program
    assert program_copy.backend is backend


@pytest.mark.parametrize(
    "backend", [gtx.embedded_numpy, gtfn.run_gtfn], ids=lambda backend: backend.name
)
def test_deepcopy_of_a_program_after_allocation(backend):
    program = _make_copy_program(backend)
    field = gtx.zeros({IDim: 3}, allocator=program)
    assert isinstance(field.ndarray, np.ndarray)

    program_copy = copy.deepcopy(program)

    assert program_copy.backend is backend
    assert "__gt_allocator__" not in program.__dict__


def test_deepcopy_of_a_default_program():
    @gtx.field_operator
    def identity(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
        return a

    @gtx.program
    def copy_program(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
        identity(a, out=out)

    assert copy_program.backend is gtx.embedded_numpy
    assert copy.deepcopy(copy_program).backend is gtx.embedded_numpy
    gtx.zeros({IDim: 3}, allocator=copy_program)
    assert copy.deepcopy(copy_program).backend is gtx.embedded_numpy


# --- Partial runs of the internal toolchain (`Toolchain.translate`) ---


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
        _make_compiled_backend(
            name="fake",
            compiler=recipes.OTFCompileWorkflow(
                translation=fake_translation,
                bindings=_unreachable_step,
                compilation=_unreachable_step,
            ),
        )
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
        _make_compiled_backend(
            name="fake",
            compiler=recipes.OTFCompileWorkflow(
                translation=lambda inp: SENTINEL_SOURCE,
                bindings=_unreachable_step,
                compilation=_unreachable_step,
            ),
        )
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
        _make_compiled_backend(name="monolithic", compiler=_unreachable_step),
        frontend=recording_frontend,
    )

    with pytest.raises(NotImplementedError, match="OTFCompileWorkflow") as exc_info:
        toolchain.translate(copy_prog.definition_stage, compile_time_args)

    message = str(exc_info.value)
    assert "monolithic" in message  # names the backend
    assert "function" in message  # names the offending compiler type
    assert frontend_calls == []  # fails fast, before running the frontend
