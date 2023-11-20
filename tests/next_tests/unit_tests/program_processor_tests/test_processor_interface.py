# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

import gt4py.next.allocators as next_allocators
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.processor_interface import (
    ProgramBackend,
    ProgramExecutor,
    ProgramFormatter,
    ProgramProcessor,
    ensure_processor_kind,
    is_processor_kind,
    is_program_backend,
    make_program_processor,
    program_formatter,
)


def test_make_program_processor(dummy_formatter):
    def my_func(program: itir.FencilDefinition, *args, **kwargs) -> None:
        return None

    processor = make_program_processor(my_func, ProgramExecutor)
    assert is_processor_kind(processor, ProgramExecutor)
    assert processor.__name__ == my_func.__name__
    assert processor(None) == my_func(None)

    def other_func(program: itir.FencilDefinition, *args, **kwargs) -> str:
        return f"{args}, {kwargs}"

    processor = make_program_processor(
        other_func, ProgramFormatter, name="new_name", accept_args=2, accept_kwargs=["a", "b"]
    )
    assert is_processor_kind(processor, ProgramFormatter)
    assert processor.__name__ == "new_name"
    assert processor(None) == other_func(None)
    assert processor(1, 2, a="A", b="B") == other_func(1, 2, a="A", b="B")
    assert processor(1, 2, 3, 4, a="A", b="B", c="C") != other_func(1, 2, 3, 4, a="A", b="B", c="C")

    with pytest.raises(ValueError, match="accepted arguments cannot be a negative number"):
        make_program_processor(my_func, ProgramFormatter, accept_args=-1)

    with pytest.raises(ValueError, match="invalid list of keyword argument names"):
        make_program_processor(my_func, ProgramFormatter, accept_kwargs=["a", None])


@pytest.fixture
def dummy_formatter():
    @program_formatter
    def dummy_formatter(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        return ""

    yield dummy_formatter


def test_decorated_formatter_function_is_recognized(dummy_formatter):
    ensure_processor_kind(dummy_formatter, ProgramFormatter)


def test_undecorated_formatter_function_is_not_recognized():
    def undecorated_formatter(fencil: itir.FencilDefinition, *args, **kwargs) -> str:
        return ""

    with pytest.raises(TypeError, match="is not a ProgramFormatter"):
        ensure_processor_kind(undecorated_formatter, ProgramFormatter)


def test_wrong_processor_type_is_caught_at_runtime(dummy_formatter):
    with pytest.raises(TypeError, match="is not a ProgramExecutor"):
        ensure_processor_kind(dummy_formatter, ProgramExecutor)


def test_is_program_backend():
    class DummyProgramExecutor(ProgramExecutor):
        def __call__(self, program: itir.FencilDefinition, *args, **kwargs) -> None:
            return None

    assert not is_program_backend(DummyProgramExecutor())

    class DummyAllocatorFactory:
        __gt_allocator__ = next_allocators.StandardCPUFieldBufferAllocator()

    assert not is_program_backend(DummyAllocatorFactory())

    class DummyBackend(DummyProgramExecutor, DummyAllocatorFactory):
        def __call__(self, program: itir.FencilDefinition, *args, **kwargs) -> None:
            return None

    assert is_program_backend(DummyBackend())
