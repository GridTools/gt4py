# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Embedded fields of torch tensors, on the CPU and on CUDA, beyond the namespace-parametrized tests."""

import operator
import subprocess
import sys

import numpy as np
import pytest

from gt4py.next import common
from gt4py.next.embedded import array_ops, context as embedded_context, operators
from gt4py.next.ffront import fbuiltins


pytestmark = pytest.mark.requires_torch

I = common.Dimension("I")
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)


@pytest.fixture
def torch():
    import torch

    return torch


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
def device(request):
    return request.param


def _field(torch, values, device, dtype=None):
    tensor = torch.asarray(values, dtype=dtype or torch.float64, device=device)
    return common._field(tensor, domain=common.domain({I: tensor.shape[0]}))


def test_torch_ops_registered_on_first_tensor():
    code = (
        "import sys, gt4py.next, torch\n"
        "from gt4py.next.embedded import array_ops\n"
        "assert 'gt4py.next.embedded.array_libraries.torch_ops' not in sys.modules\n"
        "assert type(array_ops.of(torch.empty(1))).__name__ == 'TorchOps'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_not_imported_with_gt4py():
    code = "import sys, gt4py.next\nassert 'torch' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_array_ns(torch, device):
    assert _field(torch, [1.0], device).array_ns is torch


@pytest.mark.parametrize(
    "builtin, np_func",
    [
        (fbuiltins.minimum, np.minimum),
        (fbuiltins.maximum, np.maximum),
        (fbuiltins.fmod, np.fmod),
        (operator.add, operator.add),
        (operator.lt, operator.lt),
        (operator.eq, operator.eq),
    ],
)
@pytest.mark.parametrize(
    "scalar", [0.5, np.float64(0.5), np.float32(0.5)], ids=["python", "float64", "float32"]
)
@pytest.mark.parametrize("scalar_first", [True, False])
def test_binary_builtin_with_scalar(torch, device, builtin, np_func, scalar, scalar_first):
    field = _field(torch, [1.0, -2.0, 3.0], device)
    args = (scalar, field) if scalar_first else (field, scalar)
    expected = np_func(*(a.asnumpy() if isinstance(a, common.Field) else a for a in args))

    result = builtin(*args)

    assert result.ndarray.device.type == device
    assert result.asnumpy().dtype == expected.dtype
    np.testing.assert_allclose(result.asnumpy(), expected)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_logical_op_with_python_bool(torch, device, op):
    field = _field(torch, [True, False], device, dtype=torch.bool)

    result = op(field, True)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), op(np.asarray([True, False]), True))


def test_python_int_keeps_integer_dtype(torch, device):
    field = _field(torch, [1, 2], device, dtype=torch.int32)

    assert (field + 1).ndarray.dtype == torch.int32
    assert fbuiltins.minimum(field, 1).ndarray.dtype == torch.int32


def test_where_with_scalar_condition_keeps_device(torch, device):
    result = fbuiltins.where(True, _field(torch, [1.0, 2.0], device), 0.0)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), [1.0, 2.0])


def test_scan_result_on_device_of_inputs(torch, device):
    scan = operators.ScanOperator(lambda state, x: state + x, forward=True, init=0.0, axis=K)
    x = common._field(
        torch.ones(4, dtype=torch.float64, device=device), domain=common.domain({K: 4})
    )

    with embedded_context.update(
        closure_column_range=common.NamedRange(K, common.UnitRange(0, 4)), offset_provider={}
    ):
        result = scan(x)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), [1.0, 2.0, 3.0, 4.0])


def test_asnumpy_of_tensor_requiring_grad(torch, device):
    tensor = torch.asarray([1.0, 2.0], dtype=torch.float64, device=device).requires_grad_()
    field = common._field(tensor, domain=common.domain({I: 2}))

    np.testing.assert_array_equal(field.asnumpy(), [1.0, 2.0])


def test_buffer_info_not_implemented(torch, device):
    with pytest.raises(NotImplementedError):
        _field(torch, [1.0], device).__gt_buffer_info__


def test_grad_through_premap_with_skip_values(torch, device):
    V = common.Dimension("V")
    E = common.Dimension("E")
    E2VDim = common.Dimension("E2V", kind=common.DimensionKind.LOCAL)
    e2v = common._connectivity(
        torch.asarray([[0, 1], [2, -1]], dtype=torch.int32, device=device),
        codomain=V,
        domain=common.domain({E: (0, 2), E2VDim: (0, 2)}),
        skip_value=-1,
    )
    # the skip entry gathers some element before it is masked, and the gradient is NaN where
    # the operator is singular at that element, so every element is regular for `sqrt`
    x = torch.tensor([4.0, 9.0, 16.0, 1.0], dtype=torch.float64, device=device, requires_grad=True)
    field = common._field(x, domain=common.domain({V: (0, 4)}))

    with embedded_context.update(offset_provider={"E2V": e2v}):
        result = fbuiltins.neighbor_sum(fbuiltins.sqrt(field.premap(e2v)), axis=E2VDim)
    loss = torch.sum(result.ndarray)
    loss.backward()

    np.testing.assert_allclose(loss.item(), 9.0)
    np.testing.assert_allclose(x.grad.cpu().numpy(), [0.25, 1.0 / 6.0, 0.125, 0.0])


def test_ops_of_tensor(torch, device):
    ops = array_ops.of(torch.empty(1, device=device))

    assert ops.array_ns is torch
    assert ops.device(torch.empty(1, device=device)).type == device
