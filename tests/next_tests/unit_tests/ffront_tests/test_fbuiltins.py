# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.ffront import fbuiltins
from gt4py.next.type_system import type_specifications as ts


# values inside the domain of every unary math builtin (0.5 is invalid for `arccosh`)
_SAFE_INPUT = {"arccosh": 2.0}


@pytest.mark.parametrize("tuple_spelling", [typing.Tuple, tuple, typing.Tuple[int], tuple[int]])
def test_type_conversion_helper_accepts_both_tuple_spellings(tuple_spelling):
    assert fbuiltins._type_conversion_helper(tuple_spelling) is ts.TupleType


@pytest.mark.parametrize(
    "union",
    [
        typing.Union[common.Field, typing.Tuple],
        # PEP 604 builds a 'types.UnionType', which has no '__origin__' at all
        common.Field | tuple,
    ],
)
def test_type_conversion_helper_accepts_both_union_spellings(union):
    assert fbuiltins._type_conversion_helper(union) == (ts.FieldType, ts.TupleType)


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
