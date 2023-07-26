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

from gt4py.next.ffront import fbuiltins
from gt4py.next.ffront import nd_array_field
from gt4py.next import common

from next_tests.integration_tests.feature_tests.math_builtin_test_data import math_builtin_test_data

import pytest
import math
import numpy as np


@pytest.fixture(params=nd_array_field._nd_array_implementations)
def nd_array_implementation(request):
    yield request.param


@pytest.mark.parametrize("builtin_name, inputs", math_builtin_test_data())
def test_math_function_builtins_execution(builtin_name: str, inputs, nd_array_implementation):
    if builtin_name == "gamma":
        # numpy has no gamma function
        pytest.xfail("TODO: implement gamma")
        ref_impl: Callable = np.vectorize(math.gamma)
    else:
        ref_impl: Callable = getattr(np, builtin_name)

    expected = ref_impl(*[np.asarray(inp, dtype=np.float32) for inp in inputs])

    inputs = [
        common.field(
            nd_array_implementation.asarray(inp, dtype=nd_array_implementation.float32),
            domain=((common.Dimension("foo"), common.UnitRange(0, len(inp))),),
        )
        for inp in inputs
    ]

    builtin = getattr(fbuiltins, builtin_name)
    result = builtin(*inputs)

    assert np.allclose(result.ndarray, expected)
