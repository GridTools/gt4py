# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixtures keeping the process-global compilation state from leaking between tests."""

from collections.abc import Iterator

import pytest

from gt4py.next.otf import compiled_program


def reset_ongoing_compilations() -> None:
    """Forget every in-flight compilation tracked for `gtx.wait_for_compilation()`.

    `compiled_program._ongoing_compilations` is process-global, and
    `wait_for_compilation()` drains *all* of it, which is the documented
    behaviour. A compiled program keeps its own reference to the future in
    `CompiledProgramsPool._compilation_jobs`, so dropping the entry here only
    removes the program from that global drain; a failed variant still raises
    its original error when it is called.
    """
    compiled_program._ongoing_compilations.clear()


@pytest.fixture(autouse=True)
def isolate_ongoing_compilations() -> Iterator[None]:
    """Stop a test from inheriting compilations left behind by an earlier one.

    Without this, a compilation that fails in one test stays tracked until the
    next `gtx.wait_for_compilation()` anywhere in the process, so an unrelated
    test drains it and fails with a foreign error. Which test is hit depends on
    ordering, so the failures look non-deterministic. See GitHub issue #2800.
    """
    yield
    reset_ongoing_compilations()
