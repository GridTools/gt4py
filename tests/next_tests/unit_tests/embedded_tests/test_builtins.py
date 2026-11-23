# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math

import numpy as np
import pytest

from gt4py.next import common, constructors
from gt4py.next.embedded import builtins as embedded_builtins, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


IDim = common.Dimension("IDim")

# values inside the domain of every unary math builtin (0.5 is invalid for `arccosh`)
_SAFE_INPUT = {"arccosh": 2.0}


@dataclasses.dataclass
class _Pair:
    a: common.Field[[IDim], np.float64]
    b: common.Field[[IDim], np.float64]


@dataclasses.dataclass
class _OtherPair:
    a: common.Field[[IDim], np.float64]
    b: common.Field[[IDim], np.float64]


def _field(values):
    return constructors.as_field([IDim], np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize(
    "builtin", embedded_builtins.ALL_BUILTINS, ids=lambda builtin: builtin.name
)
def test_every_builtin_has_a_field_implementation(builtin):
    assert nd_array_field.NdArrayField.__gt_builtin_func__(builtin) is not NotImplemented


def test_every_builtin_is_implemented():
    declared = {getattr(fbuiltins, name) for name in fbuiltins.FUN_BUILTIN_NAMES} | {
        getattr(experimental, name) for name in experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES
    }

    assert embedded_builtins.ALL_BUILTINS == declared
    for builtin in declared - embedded_builtins.FIELD_ONLY_BUILTINS:
        assert builtin in embedded_builtins._SCALAR_IMPLEMENTATIONS


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


def test_scalar_results():
    assert type(fbuiltins.astype(math.pi, np.float32)) is np.float32
    assert fbuiltins.abs(-1) == 1 and type(fbuiltins.abs(-1)) is int
    assert type(fbuiltins.gamma(np.float32(2.5))) is np.float32
    assert fbuiltins.minimum(1, 2.5) == 1.0
    assert fbuiltins.broadcast(2.0, (IDim,)) == 2.0
    as_float = fbuiltins.astype((1, 2.0), np.float32)
    assert as_float == (1.0, 2.0) and all(type(v) is np.float32 for v in as_float)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fbuiltins.where(True, 1.0, 2.0),
        lambda: fbuiltins.neighbor_sum(1.0, axis=IDim),
        lambda: experimental.concat_where(IDim < 1, 1.0, 2.0),
        lambda: fbuiltins.sqrt(np.ones(2)),
    ],
    ids=["where", "neighbor_sum", "concat_where", "sqrt of an array"],
)
def test_call_without_embedded_implementation(call):
    with pytest.raises(TypeError, match="has no embedded implementation"):
        call()


@pytest.mark.parametrize(
    "true_type, false_type", [(float, _Pair), (_Pair, float), (_Pair, _OtherPair)]
)
def test_where_rejects_mismatched_named_collections(true_type, false_type):
    def make(type_):
        if type_ is float:
            return 1.0
        return type_(*(_field(np.arange(3.0)) for _ in range(2)))

    mask = constructors.as_field([IDim], np.array([True, False, True]))
    with pytest.raises(ValueError, match="named collection of the same type"):
        fbuiltins.where(mask, make(true_type), make(false_type))


def test_where_rejects_mismatched_tuples():
    mask = constructors.as_field([IDim], np.array([True, False]))
    a = _field([1.0, 2.0])

    with pytest.raises(ValueError, match="Either both or none can be tuple"):
        fbuiltins.where(mask, (a, a), a)
    with pytest.raises(ValueError, match="Tuple of different size"):
        fbuiltins.where(mask, (a, a), (a,))


def test_where_and_astype_map_nested_tuples():
    mask = constructors.as_field([IDim], np.array([True, False]))
    a, b = _field([1.0, 2.0]), _field([3.0, 4.0])

    (first, (second,)) = fbuiltins.where(mask, (a, (b,)), (b, (a,)))
    np.testing.assert_array_equal(first.asnumpy(), [1.0, 4.0])
    np.testing.assert_array_equal(second.asnumpy(), [3.0, 2.0])

    (as_float, (scalar,)) = fbuiltins.astype((a, (2,)), np.float32)
    assert as_float.asnumpy().dtype == np.float32
    assert type(scalar) is np.float32
