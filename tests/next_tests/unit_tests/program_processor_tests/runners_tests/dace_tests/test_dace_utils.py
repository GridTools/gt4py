# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test utility functions of the dace backend module."""

import pytest

import dace

from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils


def test_safe_replace_symbolic():
    assert gtx_dace_utils.safe_replace_symbolic(
        dace.symbolic.pystr_to_symbolic("x*x + y"), symbol_mapping={"x": "y", "y": "x"}
    ) == dace.symbolic.pystr_to_symbolic("y*y + x")
