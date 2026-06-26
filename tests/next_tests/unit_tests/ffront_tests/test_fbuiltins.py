# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py.next.ffront import fbuiltins


# values inside the domain of every unary math builtin (0.5 is invalid for `arccosh`)
_SAFE_INPUT = {"arccosh": 2.0}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name", fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES + fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
)
def test_unary_math_builtin_scalar_preserves_dtype(name, dtype):
    value = dtype(_SAFE_INPUT.get(name, 0.5))
    assert type(getattr(fbuiltins, name)(value)) is dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES)
def test_unary_math_predicate_builtin_scalar_returns_bool(name, dtype):
    assert isinstance(getattr(fbuiltins, name)(dtype(0.5)), (bool, np.bool_))
