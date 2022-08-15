# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from functional.fencil_processors.processor_interface import (
    FencilExecutor,
    FencilFormatter,
    FencilSourceModuleGenerator,
    ensure_processor_kind,
    fencil_formatter,
)
from functional.fencil_processors.source_modules.source_modules import SourceModule
from functional.iterator.ir import FencilDefinition


@pytest.fixture
def dummy_formatter():
    @fencil_formatter
    def dummy_formatter(fencil: FencilDefinition, *args, **kwargs) -> str:
        return ""

    yield dummy_formatter


def test_decorated_formatter_function_is_recognized(dummy_formatter):
    ensure_processor_kind(dummy_formatter, FencilFormatter)


def test_custom_source_module_generator_class_is_recognized():
    class DummyFencilSourceModuleGenerator:
        @property
        def kind(self) -> type[FencilSourceModuleGenerator]:
            return FencilSourceModuleGenerator

        def __call__(self, fencil: FencilDefinition, *args, **kwargs) -> SourceModule:
            return SourceModule()

    ensure_processor_kind(DummyFencilSourceModuleGenerator(), FencilSourceModuleGenerator)


def test_undecorated_formatter_function_is_not_recognized():
    def undecorated_formatter(fencil: FencilDefinition, *args, **kwargs) -> str:
        return ""

    with pytest.raises(RuntimeError, match="is not a FencilFormatter"):
        ensure_processor_kind(undecorated_formatter, FencilFormatter)


def test_wrong_processor_type_is_caught_at_runtime(dummy_formatter):
    with pytest.raises(RuntimeError, match="is not a FencilExecutor"):
        ensure_processor_kind(dummy_formatter, FencilExecutor)
