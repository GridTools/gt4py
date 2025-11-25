# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pytest

from gt4py.eve import visitors

from .. import definitions

if TYPE_CHECKING:
    from pytest_benchmark import fixture as ptb_fixture


@pytest.mark.parametrize("width", [2, 5, 10])
@pytest.mark.parametrize("num_levels", [2, 3, 5])
def benchmark_empty_visit(benchmark: ptb_fixture.BenchmarkFixture, num_levels: int, width: int):
    class DummyVisitor(visitors.NodeVisitor): ...

    tree = definitions.make_recursive_compound_node(num_levels=num_levels, width=width)

    @benchmark
    def traverse():
        DummyVisitor().visit(tree)
