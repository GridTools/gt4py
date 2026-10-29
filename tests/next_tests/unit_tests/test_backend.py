# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend kinds and the internal toolchain in `gt4py.next.backend`."""

import dataclasses

import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, custom_layout_allocators as next_allocators
from gt4py.next.otf import arguments, artifacts


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
        next_backend.DEFAULT_FRONTEND
    )


def test_toolchain_is_not_public():
    assert "Toolchain" not in gtx.__all__
    assert "Toolchain" not in gtx.typing.__all__
    assert "Backend" in gtx.typing.__all__
