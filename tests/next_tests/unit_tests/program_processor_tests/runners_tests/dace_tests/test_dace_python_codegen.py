# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test python codegen from GTIR scalar expressions."""

import pytest
from gt4py.next.iterator.ir_utils import ir_makers as im

dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners.dace import gtir_python_codegen


@pytest.mark.parametrize(
    "param", [
        (im.tuple_get(1, im.call("get_domain")("arg", 0)), "arg_0_range_1"),
    ],
)
def test_safe_replace_symbolic(param):
    assert gtir_python_codegen.get_source(param[0]) == param[1]
