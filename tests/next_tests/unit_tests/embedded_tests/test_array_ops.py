# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import array_api_compat
import array_api_compat.numpy as compat_numpy
import array_api_strict as xs
import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.embedded import array_libraries, array_ops, nd_array_field
from gt4py.next.ffront import fbuiltins

from next_tests.unit_tests.conftest import asarray, strict_array_on_device1


D = common.Dimension("D")


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(array_ops, "_ops_by_namespace", dict(array_ops._ops_by_namespace))
    monkeypatch.setattr(array_ops, "_ops_by_array_type", {})


def _strict_field(values, dtype=xs.float64):
    return common._field(xs.asarray(values, dtype=dtype), domain=common.domain({D: len(values)}))


def test_unregistered_namespace_uses_standard_ops():
    field = _strict_field([1.0, 2.0])

    assert type(array_ops.of(field.ndarray)) is array_ops.ArrayOps
    assert field.array_ns is xs
    assert isinstance(field, nd_array_field.NdArrayField)


def test_registered_namespaces():
    assert type(array_ops.of(np.empty(1))) is array_libraries.numpy_ops.NumPyOps
    assert array_ops.of(np.empty(1)).array_ns is np


def test_elementwise_function_override(isolated_registry):
    class Ops(array_ops.ArrayOps):
        def _elementwise_function(self, name):
            if name == "sin":
                return lambda x: x + 100.0
            return super()._elementwise_function(name)

    array_ops.register(Ops(xs))
    field = _strict_field([1.0, 2.0])

    assert np.allclose(fbuiltins.sin(field).asnumpy(), [101.0, 102.0])
    assert np.allclose(fbuiltins.cos(field).asnumpy(), np.cos([1.0, 2.0]))


def test_builtin_override(isolated_registry):
    def where_override(cond, x, y):
        return "overridden"

    class Ops(array_ops.ArrayOps):
        builtin_overrides = {fbuiltins.where: where_override}

    array_ops.register(Ops(xs))
    field = _strict_field([1.0, 2.0])
    cond = _strict_field([True, False], dtype=xs.bool)

    assert fbuiltins.where(cond, field, field) == "overridden"
    assert np.allclose(fbuiltins.minimum(field, 1.5).asnumpy(), [1.0, 1.5])


def test_later_registration_takes_precedence(isolated_registry):
    class Ops(array_libraries.numpy_ops.NumPyOps):
        pass

    array_ops.register(Ops(compat_numpy))

    assert type(array_ops.of(np.empty(1))) is Ops


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_empty(nd_array_implementation, dtype):
    ops = array_ops.of(asarray(nd_array_implementation, [1.0]))

    result = ops.empty((2, 3), dtype=dtype)

    assert array_api_compat.array_namespace(result) is ops.xp
    assert result.shape == (2, 3)
    assert ops.scalar_type(result.dtype) is dtype


def test_empty_on_device_of_like(nd_array_implementation):
    like = asarray(nd_array_implementation, [1.0])
    ops = array_ops.of(like)

    result = ops.empty((2,), dtype=np.float64, like=like)

    assert array_api_compat.array_namespace(result) is ops.xp
    assert array_api_compat.device(result) == array_api_compat.device(like)


@pytest.mark.requires_torch
@pytest.mark.requires_torch_cuda
@pytest.mark.requires_gpu
def test_empty_on_cuda_device_of_torch_like():
    import torch

    like = torch.ones(1, device="cuda")

    result = array_ops.of(like).empty((2,), dtype=np.float64, like=like)

    assert result.device == like.device


def test_empty_on_non_default_device_of_like():
    like = strict_array_on_device1([1.0])

    result = array_ops.of(like).empty((2,), dtype=np.float64, like=like)

    assert result.device == like.device != xs.empty((1,)).device
