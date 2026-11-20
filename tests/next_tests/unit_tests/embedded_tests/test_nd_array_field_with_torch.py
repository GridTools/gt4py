# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import operator

import numpy as np
import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.common import Dimension, DimensionKind, NamedRange, UnitRange
from gt4py.next.ffront import fbuiltins


D0 = Dimension("D0")
D1 = Dimension("D1")
D2 = Dimension("D2")

_TORCH_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=(pytest.mark.requires_gpu, pytest.mark.requires_torch_cuda)),
]


def _torch_field(values, device, dtype=None):
    import torch

    tensor = torch.asarray(values, dtype=dtype or torch.float64, device=device)
    return common._field(tensor, domain=common.domain({D0: tensor.shape[0]}))


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
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
def test_torch_binary_builtin_with_scalar(device, builtin, np_func, scalar, scalar_first):
    field = _torch_field([1.0, -2.0, 3.0], device)
    args = (scalar, field) if scalar_first else (field, scalar)
    expected = np_func(*(a.asnumpy() if isinstance(a, common.Field) else a for a in args))

    result = builtin(*args)

    assert result.ndarray.device.type == device
    assert result.asnumpy().dtype == expected.dtype
    np.testing.assert_allclose(result.asnumpy(), expected)


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_torch_logical_op_with_python_bool(device, op):
    import torch

    field = _torch_field([True, False], device, dtype=torch.bool)

    result = op(field, True)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), op(np.asarray([True, False]), True))


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_python_int_keeps_integer_dtype(device):
    import torch

    field = _torch_field([1, 2], device, dtype=torch.int32)

    assert (field + 1).ndarray.dtype == torch.int32
    assert fbuiltins.minimum(field, 1).ndarray.dtype == torch.int32


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_where_with_scalar_condition_keeps_device(device):
    result = fbuiltins.where(True, _torch_field([1.0, 2.0], device), 0.0)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), [1.0, 2.0])


@pytest.mark.requires_torch
def test_torch_gamma_at_poles_matches_scipy():
    import scipy.special

    x = [0.0, -0.0, -1.0, -2.0, -2.5, 0.5, 5.0]

    result = fbuiltins.gamma(_torch_field(x, "cpu"))

    np.testing.assert_allclose(result.asnumpy(), scipy.special.gamma(x), equal_nan=True)


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_scan_result_on_device_of_inputs(device):
    import torch

    from gt4py.next.embedded import context as embedded_context, operators

    K = Dimension("K", kind=DimensionKind.VERTICAL)
    scan = operators.ScanOperator(lambda state, x: state + x, forward=True, init=0.0, axis=K)
    x = common._field(
        torch.ones(4, dtype=torch.float64, device=device), domain=common.domain({K: 4})
    )

    with embedded_context.update(
        closure_column_range=NamedRange(K, UnitRange(0, 4)), offset_provider={}
    ):
        result = scan(x)

    assert result.ndarray.device.type == device
    np.testing.assert_array_equal(result.asnumpy(), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_asnumpy_of_tensor_requiring_grad(device):
    import torch

    tensor = torch.asarray([1.0, 2.0], dtype=torch.float64, device=device).requires_grad_()
    field = common._field(tensor, domain=common.domain({D0: 2}))

    np.testing.assert_array_equal(field.asnumpy(), [1.0, 2.0])


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_grad_through_premap_with_skip_values(device):
    import torch

    from gt4py.next.embedded import context as embedded_context

    V = Dimension("V")
    E = Dimension("E")
    E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
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


@pytest.mark.requires_torch
@pytest.mark.parametrize("device", _TORCH_DEVICES)
def test_torch_buffer_info(device):
    import torch

    data = torch.arange(24.0, dtype=torch.float64, device=device).reshape(2, 3, 4)[:, 1:, ::2]
    field = common._field(data, domain=common.domain({D0: 2, D1: 2, D2: 2}))

    buffer_info = field.__gt_buffer_info__

    assert buffer_info.data_ptr == data.data_ptr()
    assert buffer_info.shape == tuple(data.shape)
    assert buffer_info.elem_strides == data.stride()
    assert buffer_info.device.device_type == (
        core_defs.DeviceType.CUDA if device == "cuda" else core_defs.DeviceType.CPU
    )
