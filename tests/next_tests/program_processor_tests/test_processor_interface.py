# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.processor_interface import (
    ProgramExecutor,
    ProgramFormatter,
    ensure_processor_kind,
    program_formatter,
)


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
