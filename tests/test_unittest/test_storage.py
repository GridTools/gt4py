# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from types import SimpleNamespace

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest


try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py.backend as gt_backend
import gt4py.storage as gt_store
import gt4py.storage.utils as gt_storage_utils
from gt4py.gtscript import PARALLEL, Field, computation, interval, stencil
from gt4py.storage.storage import GPUStorage

from ..definitions import CPU_BACKENDS, GPU_BACKENDS


try:
    import dace
except ImportError:
    dace = None

# ---- Hypothesis strategies ----
@hyp_st.composite
def allocation_strategy(draw):
    dtype = np.dtype(
        draw(
            hyp_st.one_of(
                list(
                    hyp_st.just(d)
                    for d in [
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                        np.float16,
                        np.float32,
                        np.float64,
                    ]
                )
            )
        )
    )
    alignment = draw(hyp_st.integers(min_value=1, max_value=64)) * dtype.itemsize
    dimension = draw(hyp_st.integers(min_value=1, max_value=4))

    shape_strats = []
    default_origin_strats = []

    for i in range(dimension):
        shape_strats = shape_strats + [hyp_st.integers(min_value=1, max_value=64)]
    shape = draw(hyp_st.tuples(*shape_strats))
    for i in range(dimension):
        default_origin_strats = default_origin_strats + [
            hyp_st.integers(min_value=0, max_value=min(32, shape[i] - 1))
        ]
    default_origin = draw(hyp_st.tuples(*default_origin_strats))
    layout_order = draw(hyp_st.permutations(tuple(range(dimension))))
    return dict(
        dtype=dtype,
        alignment=alignment,
        shape=shape,
        default_origin=default_origin,
        layout_order=layout_order,
    )


@hyp_st.composite
def mask_strategy(draw):
    dimension = draw(hyp_st.integers(min_value=1, max_value=6))

    shape_strats = [hyp_st.integers(min_value=1, max_value=32)] * dimension
    shape = draw(hyp_st.tuples(*shape_strats))
    default_origin_strats = [
        hyp_st.integers(min_value=0, max_value=min(32, shape[i] - 1)) for i in range(dimension)
    ]
    default_origin = draw(hyp_st.tuples(*default_origin_strats))

    mask_values = [True] * dimension
    if dimension < 3:
        mask_values += [False] * (3 - dimension)

    mask = draw(
        hyp_st.one_of(
            hyp_st.just(None),
            hyp_st.permutations(mask_values),
        )
    )
    return dict(shape=shape, default_origin=default_origin, mask=mask)


# ---- Tests ----
@hyp.given(param_dict=allocation_strategy())
def test_allocate_cpu(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]

    raw_buffer, field = gt_storage_utils.allocate_cpu(
        default_origin, shape, layout_map, dtype, alignment_bytes
    )

    # check that field is a view of raw_buffer
    assert field.base is raw_buffer

    # check that memory of field is contained in raw_buffer
    assert (
        field.ctypes.data >= raw_buffer.ctypes.data
        and field[-1:].ctypes.data <= raw_buffer[-1:].ctypes.data
    )

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    for i in range(100):
        slices = []
        for hidx in range(len(shape)):
            if hidx == np.argmax(layout_map):
                slices = slices + [slice(default_origin[hidx], None)]
            else:
                slices = slices + [slice(random.randint(0, shape[hidx]), None)]
        assert field[tuple(slices)].ctypes.data % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    slices = []
    for hidx in range(len(shape)):
        slices = slices + [0]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [default_origin[hidx]]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx] - 1]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx]]
    try:
        field[tuple(slices)] = 1
    except IndexError:
        pass
    else:
        assert False

    # check if shape is properly set
    assert field.shape == shape


@pytest.mark.requires_gpu
@hyp.given(param_dict=allocation_strategy())
def test_allocate_gpu(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]

    raw_buffer, field = gt_storage_utils.allocate_gpu(
        default_origin, shape, layout_map, dtype, alignment_bytes
    )

    # check that memory of field is contained in raw_buffer
    assert (
        field.ctypes.data >= raw_buffer.data.ptr
        and field[-1:].ctypes.data <= raw_buffer[-1:].data.ptr
    )

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    for i in range(100):
        slices = []
        for hidx in range(len(shape)):
            if hidx == np.argmax(layout_map):
                slices = slices + [slice(default_origin[hidx], None)]
            else:
                slices = slices + [slice(random.randint(0, shape[hidx]), None)]
        assert field[tuple(slices)].ctypes.data % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    slices = []
    for hidx in range(len(shape)):
        slices = slices + [0]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [default_origin[hidx]]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx] - 1]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx]]
    try:
        field[tuple(slices)] = 1
    except IndexError:
        pass
    else:
        assert False

    # check if shape is properly set
    assert field.shape == shape


@pytest.mark.requires_gpu
@hyp.given(param_dict=allocation_strategy())
def test_allocate_gpu_unmanaged(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]
    raw_buffer, field, device_raw_buffer, device_field = gt_storage_utils.allocate_gpu_unmanaged(
        default_origin, shape, layout_map, dtype, alignment_bytes
    )

    # check that field is a view of raw_buffer
    assert field.base is raw_buffer
    assert (
        field.ctypes.data >= raw_buffer.ctypes.data
        and field[-1:].ctypes.data <= raw_buffer[-1:].ctypes.data
    )

    # assert (device_field.base is device_raw_buffer) # as_strided actually returns an ndarray where base=None??
    # instead check that the memory of field is contained in raws buffer:
    assert (
        device_field.data.ptr >= device_raw_buffer.data.ptr
        and device_field[-1:].data.ptr <= device_raw_buffer[-1:].data.ptr
    )

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    for i in range(100):
        slices = []
        for hidx in range(len(shape)):
            if hidx == np.argmax(layout_map):
                slices = slices + [slice(default_origin[hidx], None)]
            else:
                slices = slices + [slice(random.randint(0, shape[hidx]), None)]
        assert field[tuple(slices)].ctypes.data % alignment_bytes == 0
        assert device_field[tuple(slices)].data.ptr % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    slices = []
    for hidx in range(len(shape)):
        slices = slices + [0]
    field[tuple(slices)] = 1
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [default_origin[hidx]]
    field[tuple(slices)] = 1
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx] - 1]
    field[tuple(slices)] = 1
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx]]
    try:
        field[tuple(slices)] = 1
    except IndexError:
        pass
    else:
        assert False
    try:
        device_field[tuple(slices)] = 1
    except IndexError:
        pass
    else:
        assert False

    # check if shape is properly set
    assert field.shape == shape
    assert device_field.shape == shape


def test_normalize_storage_spec():
    from gt4py import gtscript
    from gt4py.storage.utils import normalize_storage_spec

    default_origin = (0, 0, 0)
    shape = (10, 10, 10)
    dtype = np.float64
    mask = (True, True, True)

    default_origin_out, shape_out, dtype_out, mask_out = normalize_storage_spec(
        default_origin, shape, dtype, mask
    )
    assert default_origin_out == default_origin
    assert shape_out == shape
    assert dtype_out == dtype
    assert mask_out == mask

    # Default origin
    default_origin_out, shape_out, dtype_out, mask_out = normalize_storage_spec(
        (1, 1, 1), shape, dtype, mask
    )
    assert default_origin_out == (1, 1, 1)
    assert shape_out == shape
    assert dtype_out == dtype
    assert mask_out == mask

    with pytest.raises(TypeError, match="default_origin"):
        normalize_storage_spec(None, shape, dtype, mask)
    with pytest.raises(TypeError, match="default_origin"):
        normalize_storage_spec(("1", "1", "1"), shape, dtype, mask)
    with pytest.raises(ValueError, match="default_origin"):
        normalize_storage_spec((1, 1, 1, 1), shape, dtype, mask)
    with pytest.raises(ValueError, match="default_origin"):
        normalize_storage_spec((-1, -1, -1), shape, dtype, mask)

    # Shape
    default_origin_out, shape_out, dtype_out, mask_out = normalize_storage_spec(
        default_origin, (10, 10), dtype, (True, True, False)
    )
    assert default_origin_out == (0, 0)
    assert shape_out == (10, 10)
    assert dtype_out == dtype
    assert mask_out == (True, True, False)

    with pytest.raises(ValueError, match="non-matching"):
        normalize_storage_spec(default_origin, (10, 20), dtype, (False, True, False))
    with pytest.raises(TypeError, match="shape"):
        normalize_storage_spec(default_origin, "(10,10)", dtype, mask)
    with pytest.raises(TypeError, match="shape"):
        normalize_storage_spec(default_origin, None, dtype, mask)
    with pytest.raises(ValueError, match="shape"):
        normalize_storage_spec(default_origin, (10, 10, 0), dtype, mask)

    # Mask
    default_origin_out, shape_out, dtype_out, mask_out = normalize_storage_spec(
        default_origin, (10, 10), dtype, (True, True, False)
    )
    assert default_origin_out == (0, 0)
    assert shape_out == (10, 10)
    assert dtype_out == dtype
    assert mask_out == (True, True, False)

    _, __, ___, mask_out = normalize_storage_spec(default_origin, (10, 10), dtype, "IJ")
    assert mask_out == (True, True, False)

    _, __, ___, mask_out = normalize_storage_spec(default_origin, (10, 10), dtype, gtscript.IJ)
    assert mask_out == (True, True, False)

    _, __, ___, mask_out = normalize_storage_spec(default_origin, (10, 10, 10), dtype, gtscript.IJK)
    assert mask_out == (True, True, True)

    with pytest.raises(ValueError, match="mask"):
        normalize_storage_spec(default_origin, (10, 10), dtype, (False, False, False))

    # Dtype
    default_origin_out, shape_out, dtype_out, mask_out = normalize_storage_spec(
        default_origin, shape, (dtype, (2,)), mask
    )
    assert default_origin_out == tuple([*default_origin, 0])
    assert shape_out == tuple([*shape, 2])
    assert dtype_out == dtype
    assert mask_out == tuple([*mask, True])


@pytest.mark.parametrize(
    "alloc_fun",
    [
        gt_store.empty,
        gt_store.ones,
        gt_store.zeros,
        lambda dtype, default_origin, shape, backend: gt_store.from_array(
            np.empty(shape, dtype=dtype),
            backend=backend,
            shape=shape,
            dtype=dtype,
            default_origin=default_origin,
        ),
    ],
)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_cpu_constructor(alloc_fun, backend):
    stor = alloc_fun(dtype=np.float64, default_origin=(1, 2, 3), shape=(2, 4, 6), backend=backend)
    assert type(stor.default_origin) is tuple
    assert stor.default_origin == (1, 2, 3)
    assert type(stor.shape) is tuple
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, np.ndarray)
    assert stor.is_stencil_view


@pytest.mark.parametrize(
    "alloc_fun",
    [
        gt_store.empty,
        gt_store.ones,
        gt_store.zeros,
        lambda dtype, default_origin, shape, backend: gt_store.from_array(
            np.empty(shape, dtype=dtype),
            backend=backend,
            shape=shape,
            dtype=dtype,
            default_origin=default_origin,
        ),
    ],
)
@pytest.mark.parametrize(
    "backend",
    GPU_BACKENDS,
)
def test_gpu_constructor(alloc_fun, backend):
    stor = alloc_fun(dtype=np.float64, default_origin=(1, 2, 3), shape=(2, 4, 6), backend=backend)
    assert type(stor.default_origin) is tuple
    assert stor.default_origin == (1, 2, 3)
    assert type(stor.shape) is tuple
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, np.ndarray)
    assert stor.is_stencil_view


@hyp.given(param_dict=mask_strategy())
def test_masked_storage_cpu(param_dict):
    mask = param_dict["mask"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]

    # no assert when all is defined in descriptor, no grid_group
    store = gt_store.empty(
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        mask=mask,
        backend="gt:cpu_kfirst",
    )
    assert sum(store.mask) == store.ndim
    assert sum(store.mask) == len(store.data.shape)


@pytest.mark.requires_gpu
@hyp.given(param_dict=mask_strategy())
def test_masked_storage_gpu(param_dict):
    mask = param_dict["mask"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]

    # no assert when all is defined in descriptor, no grid_group
    store = gt_store.empty(
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        mask=mask,
        backend="gt:gpu",
    )
    assert sum(store.mask) == store.ndim
    assert sum(store.mask) == len(store.data.shape)


def test_masked_storage_asserts():
    default_origin = (1, 1, 1)
    shape = (2, 2, 2)
    backend = "gt:cpu_kfirst"

    with pytest.raises(ValueError):
        gt_store.empty(
            dtype=np.float64,
            default_origin=default_origin,
            shape=shape,
            mask=(),
            backend=backend,
        )


def run_test_slices(backend):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = np.random.randn(*shape)
    stor = gt_store.from_array(
        array, backend=backend, dtype=np.float64, default_origin=default_origin, shape=shape
    )
    sliced = stor[::2, ::2, ::2]
    assert (sliced.view(np.ndarray) == array[::2, ::2, ::2]).all()
    sliced[...] = array[::2, ::2, ::2]


def test_slices_cpu():
    run_test_slices(backend="gt:cpu_ifirst")


@pytest.mark.requires_gpu
def test_slices_gpu():
    run_test_slices(backend="gt:gpu")

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array, backend="gt:gpu", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    sliced = stor[::2, ::2, ::2]
    # assert (sliced == array[::2, ::2, ::2]).all()
    sliced[...] = array[::2, ::2, ::2]

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array, backend="gt:gpu", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    ref = gt_store.from_array(
        array, backend="gt:gpu", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import copy  # isort:skip
    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array, backend="gt:gpu", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    ref = copy.deepcopy(stor)
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array,
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=True,
    )
    ref = gt_store.from_array(
        array,
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=True,
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import copy  # isort:skip
    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array,
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=True,
    )
    ref = copy.deepcopy(stor)
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_store.from_array(
        array,
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=False,
    )
    ref = gt_store.from_array(
        array,
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=False,
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2] + ref[::2, ::2, ::2]


def test_transpose(backend="numpy"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = np.random.randn(*shape)
    stor = gt_store.from_array(
        array, default_origin=default_origin, backend=backend, dtype=np.float64
    )
    transposed = np.transpose(stor, axes=(0, 1, 2))
    assert transposed.strides == stor.strides
    assert transposed.is_stencil_view
    transposed = np.transpose(stor, axes=(2, 1, 0))
    assert not transposed.is_stencil_view


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize(
    "method",
    ["deepcopy", "copy_method"],
)
def test_copy_cpu(method, backend):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_store.from_array(
        np.random.randn(*shape), default_origin=default_origin, backend=backend
    )

    import copy

    if method == "deepcopy":
        stor_copy = copy.deepcopy(stor)
    elif method == "copy_method":
        stor_copy = stor.copy()
    else:
        raise ValueError(f"Test not implemented for copying using '{method}'")

    assert stor is not stor_copy
    assert stor._raw_buffer.ctypes.data != stor_copy._raw_buffer.ctypes.data
    if stor._raw_buffer.ctypes.data < stor_copy._raw_buffer.ctypes.data:
        assert (
            stor._raw_buffer.ctypes.data + len(stor._raw_buffer)
            <= stor_copy._raw_buffer.ctypes.data
        )
    else:
        assert (
            stor._raw_buffer.ctypes.data + len(stor._raw_buffer)
            >= stor_copy._raw_buffer.ctypes.data
        )
    np.testing.assert_equal(stor_copy.view(np.ndarray), stor.view(np.ndarray))


@pytest.mark.requires_gpu
@pytest.mark.parametrize("method", ["deepcopy", "copy_method"])
def test_copy_gpu(method, backend="gt:gpu"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_store.from_array(
        np.random.randn(*shape),
        default_origin=default_origin,
        backend=backend,
        managed_memory=True,
    )

    import copy

    if method == "deepcopy":
        stor_copy = copy.deepcopy(stor)
    elif method == "copy_method":
        stor_copy = stor.copy()
    else:
        raise ValueError(f"Test not implemented for copying using '{method}'")

    assert stor is not stor_copy
    assert stor._raw_buffer.data.ptr != stor_copy._raw_buffer.data.ptr
    if stor._raw_buffer.data.ptr < stor_copy._raw_buffer.data.ptr:
        assert stor._raw_buffer.data.ptr + len(stor._raw_buffer) <= stor_copy._raw_buffer.data.ptr
    else:
        assert stor._raw_buffer.data.ptr + len(stor._raw_buffer) >= stor_copy._raw_buffer.data.ptr
    np.testing.assert_equal(stor_copy.view(np.ndarray), stor.view(np.ndarray))


@pytest.mark.requires_gpu
@pytest.mark.parametrize("method", ["deepcopy", "copy_method"])
def test_deepcopy_gpu_unmanaged(method, backend="gt:gpu"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_store.from_array(
        np.random.randn(*shape),
        default_origin=default_origin,
        backend=backend,
        managed_memory=False,
    )

    import copy

    if method == "deepcopy":
        stor_copy = copy.deepcopy(stor)
    elif method == "copy_method":
        stor_copy = stor.copy()
    else:
        raise ValueError(f"Test not implemented for copying using '{method}'")

    assert stor is not stor_copy
    assert stor._sync_state is not stor_copy._sync_state
    assert stor._raw_buffer.ctypes.data != stor_copy._raw_buffer.ctypes.data
    if stor._raw_buffer.ctypes.data < stor_copy._raw_buffer.ctypes.data:
        assert (
            stor._raw_buffer.ctypes.data + len(stor._raw_buffer)
            <= stor_copy._raw_buffer.ctypes.data
        )
    else:
        assert (
            stor._raw_buffer.ctypes.data + len(stor._raw_buffer)
            >= stor_copy._raw_buffer.ctypes.data
        )

    assert stor._device_raw_buffer.data.ptr != stor_copy._device_raw_buffer.data.ptr
    if stor._device_raw_buffer.data.ptr < stor_copy._device_raw_buffer.data.ptr:
        assert (
            stor._device_raw_buffer.data.ptr + len(stor._device_raw_buffer)
            <= stor_copy._device_raw_buffer.data.ptr
        )
    else:
        assert (
            stor._device_raw_buffer.data.ptr + len(stor._device_raw_buffer)
            >= stor_copy._device_raw_buffer.data.ptr
        )

    np.testing.assert_equal(stor_copy.view(np.ndarray), stor.view(np.ndarray))
    assert (stor._device_field[...] == stor_copy._device_field[...]).all()


def run_test_view(backend):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_store.from_array(
        np.random.randn(*shape), default_origin=default_origin, backend=backend
    )
    stor.view(type(stor))
    if gt_backend.from_name(backend).storage_info["layout_map"]([True] * 3) != (0, 1, 2):
        try:
            np.ones((10, 10, 10)).view(type(stor))
        except RuntimeError:
            pass
        except Exception as e:
            raise e
        else:
            raise Exception
        tmp_view = stor[::2, ::2, ::2]
        assert not tmp_view._is_consistent(stor)
        assert not tmp_view.is_stencil_view


@pytest.mark.parametrize(
    "backend",
    [
        name
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "cpu"
    ],
)
def test_view_cpu(backend):
    run_test_view(backend=backend)


@pytest.mark.requires_gpu
def test_view_gpu():
    run_test_view(backend="gt:gpu")


class TestNumpyPatch:
    def test_asarray(self):
        storage = gt_store.from_array(
            np.random.randn(5, 5, 5), default_origin=(1, 1, 1), backend="gt:cpu_ifirst"
        )

        class NDArraySub(np.ndarray):
            pass

        numpy_array = np.ones((3, 3, 3))
        matrix = np.ones((3, 3)).view(NDArraySub)

        assert isinstance(np.asarray(storage), np.ndarray)
        assert isinstance(np.asarray(numpy_array), np.ndarray)
        assert isinstance(np.asarray(matrix), np.ndarray)
        assert isinstance(np.asanyarray(storage), type(storage))
        assert isinstance(np.asanyarray(numpy_array), np.ndarray)
        assert isinstance(np.asanyarray(matrix), NDArraySub)
        assert isinstance(np.array(storage), np.ndarray)
        assert isinstance(np.array(matrix), np.ndarray)
        assert isinstance(np.array(numpy_array), np.ndarray)

        # apply numpy patch
        gt_store.prepare_numpy()

        try:
            assert isinstance(np.asarray(storage), type(storage))
            assert isinstance(np.asarray(numpy_array), np.ndarray)
            assert isinstance(np.asarray(matrix), np.ndarray)

            assert isinstance(np.array(matrix), np.ndarray)
            assert isinstance(np.array(numpy_array), np.ndarray)
            with pytest.raises(RuntimeError):
                np.array(storage)
        finally:
            # undo patch
            gt_store.restore_numpy()

        assert isinstance(np.asarray(storage), np.ndarray)
        assert isinstance(np.asarray(numpy_array), np.ndarray)
        assert isinstance(np.asarray(matrix), np.ndarray)

        assert isinstance(np.array(storage), np.ndarray)
        assert isinstance(np.array(matrix), np.ndarray)
        assert isinstance(np.array(numpy_array), np.ndarray)

    def test_array(self):
        storage = gt_store.from_array(
            np.random.randn(5, 5, 5), default_origin=(1, 1, 1), backend="gt:cpu_ifirst"
        )

        class NDArraySub(np.ndarray):
            pass

        numpy_array = np.ones((3, 3, 3))
        matrix = np.ones((3, 3)).view(NDArraySub)

        assert isinstance(np.array(storage, copy=False), np.ndarray)
        assert isinstance(np.array(numpy_array, copy=False), np.ndarray)
        assert isinstance(np.array(matrix, copy=False), np.ndarray)
        assert isinstance(np.asanyarray(storage), type(storage))
        assert isinstance(np.asanyarray(numpy_array), np.ndarray)
        assert isinstance(np.asanyarray(matrix), NDArraySub)
        assert isinstance(np.array(storage), np.ndarray)
        assert isinstance(np.array(matrix), np.ndarray)
        assert isinstance(np.array(numpy_array), np.ndarray)

        # apply numpy patch
        gt_store.prepare_numpy()
        try:
            assert isinstance(np.array(storage, copy=False), type(storage))
            assert isinstance(np.array(numpy_array, copy=False), np.ndarray)
            assert isinstance(np.array(matrix, copy=False), np.ndarray)

            assert isinstance(np.array(matrix), np.ndarray)
            assert isinstance(np.array(numpy_array), np.ndarray)
            with pytest.raises(RuntimeError):
                np.array(storage)
        finally:
            # undo patch
            gt_store.restore_numpy()

        assert isinstance(np.array(storage, copy=False), np.ndarray)
        assert isinstance(np.array(numpy_array, copy=False), np.ndarray)
        assert isinstance(np.array(matrix, copy=False), np.ndarray)

        assert isinstance(np.array(storage), np.ndarray)
        assert isinstance(np.array(matrix), np.ndarray)
        assert isinstance(np.array(numpy_array), np.ndarray)


@pytest.mark.requires_gpu
def test_cuda_array_interface():
    storage = gt_store.from_array(
        cp.random.randn(5, 5, 5),
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=(1, 1, 1),
        shape=(5, 5, 5),
    )
    cupy_array = cp.array(storage)
    assert (cupy_array == storage).all()


@pytest.mark.requires_gpu
def test_view_casting():
    cp.cuda.set_allocator(cp.cuda.malloc_managed)
    gpu_arr = cp.empty((5, 5, 5))

    gpu_arr = cp.ones((10, 10, 10))
    cpu_view = gt_storage_utils.cpu_view(gpu_arr)
    assert cpu_view.ctypes.data == gpu_arr.data.ptr
    assert cpu_view.strides == gpu_arr.strides
    assert cpu_view.shape == gpu_arr.shape

    gpu_view = gt_storage_utils.gpu_view(cpu_view)
    assert gpu_view.data.ptr == cpu_view.ctypes.data
    assert gpu_view.strides == cpu_view.strides
    assert gpu_view.shape == cpu_view.shape


@pytest.mark.requires_gpu
def test_managed_memory():
    cp.cuda.set_allocator(cp.cuda.malloc_managed)

    gpu_arr = cp.ones((10, 10, 10))
    cpu_view = gt_storage_utils.cpu_view(gpu_arr)

    gpu_arr[0, 0, 0] = 123
    cp.cuda.runtime.deviceSynchronize()
    assert cpu_view[0, 0, 0] == 123
    cpu_view[1, 1, 1] = 321
    assert gpu_arr[1, 1, 1] == 321


@pytest.mark.requires_gpu
def test_sum_gpu():
    i1 = 3
    i2 = 4
    jslice = slice(3, 4, None)
    shape = (5, 5, 5)
    q1 = gt_store.from_array(
        cp.zeros(shape),
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=(0, 0, 0),
        shape=shape,
    )

    q2 = gt_store.from_array(
        cp.ones(shape),
        backend="gt:gpu",
        dtype=np.float64,
        default_origin=(0, 0, 0),
        shape=shape,
    )

    q1[i1 : i2 + 1, jslice, 0] = cp.sum(q2[i1 : i2 + 1, jslice, :], axis=2)


@pytest.mark.requires_gpu
def test_auto_sync_storage():

    # make sure no storages are modified to begin with, e.g. by other tests.
    cp.cuda.Device(0).synchronize()
    GPUStorage._modified_storages.clear()

    BACKEND = "gt:gpu"

    @stencil(backend=BACKEND, device_sync=False)
    def swap_stencil(
        inp: Field[float],  # type: ignore
        out: Field[float],  # type: ignore
    ):
        with computation(PARALLEL), interval(...):
            tmp = inp
            inp = out
            out = tmp

    shape = (5, 5, 5)
    q0 = gt_store.from_array(
        cp.zeros(shape),
        backend=BACKEND,
        dtype=np.float64,
        default_origin=(0, 0, 0),
        shape=shape,
        managed_memory=True,
    )

    q1 = gt_store.from_array(
        cp.ones(shape),
        backend=BACKEND,
        dtype=np.float64,
        default_origin=(0, 0, 0),
        shape=shape,
        managed_memory=True,
    )
    q0_view = q0[3:, 3:, 3:]

    assert not gt_store.storage.GPUStorage.get_modified_storages()

    # call stencil and mark original storage clean
    swap_stencil(q0, q1)
    assert len(gt_store.storage.GPUStorage.get_modified_storages()) == 2
    assert q0._is_device_modified
    assert q0_view._is_device_modified
    q0_view._set_clean()
    assert len(gt_store.storage.GPUStorage.get_modified_storages()) == 1
    assert not q0._is_device_modified
    assert not q0_view._is_device_modified

    # call stencil and mark original storage clean
    swap_stencil(q0, q1)
    assert len(gt_store.storage.GPUStorage.get_modified_storages()) == 2
    assert q0._is_device_modified
    assert q0_view._is_device_modified
    q0_view._set_clean()
    assert len(gt_store.storage.GPUStorage.get_modified_storages()) == 1
    assert not q0._is_device_modified
    assert not q0_view._is_device_modified

    # call stencil and mark original storage clean
    swap_stencil(q0, q1)
    q0.device_to_host()
    assert not gt_store.storage.GPUStorage.get_modified_storages()


@pytest.mark.requires_gpu
def test_slice_gpu():
    stor = gt_store.ones(
        backend="gt:gpu",
        managed_memory=False,
        shape=(10, 10, 10),
        default_origin=(0, 0, 0),
        dtype=np.float64,
    )
    stor.synchronize()
    view = stor[1:-1, 1:-1, 1:-1]

    gpu_stor = stor._device_field
    gpu_view = view._device_field

    view_start = gpu_view.data.ptr
    storage_start = gpu_stor.data.ptr

    view_end = gpu_view[-1:, -1:, -1:].data.ptr
    storage_end = gpu_stor[-1:, -1:, -1:].data.ptr

    assert view_start > storage_start
    assert view_end < storage_end


def test_non_existing_backend():
    with pytest.raises(RuntimeError, match="backend"):
        gt_store.empty(
            "non_existing_backend",
            default_origin=[0, 0, 0],
            shape=[10, 10, 10],
            dtype=(np.float64, (3,)),
        )


@pytest.mark.skipif(dace is None, reason="Storage __descriptor__ depends on dace.")
class TestDescriptor:
    @staticmethod
    def ravel_with_padding(array):

        if hasattr(array, "__cuda_array_interface__"):
            interface = dict(array.__cuda_array_interface__)
        else:
            interface = dict(array.__array_interface__)

        assert interface.get("offset", 0) == 0 and interface["data"] is not None

        total_size = int(
            int(np.array([array.shape]) @ np.array([array.strides]).T) // array.itemsize
        )
        interface["shape"] = (total_size,)
        interface["strides"] = (min(array.strides),)

        if hasattr(array, "__cuda_array_interface__"):
            return cp.asarray(SimpleNamespace(__cuda_array_interface__=interface))
        else:
            return np.asarray(SimpleNamespace(__array_interface__=interface))

    @pytest.mark.parametrize(
        "backend",
        [
            "dace:cpu",
            pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu]),
        ],
    )
    def test_device(self, backend):
        backend_cls = gt_backend.from_name(backend)
        stor = gt_store.ones(
            backend=backend,
            managed_memory=False,
            shape=(3, 7, 13),
            default_origin=(1, 2, 3),
            dtype=np.float64,
        )
        descriptor: dace.data.Array = stor.__descriptor__()
        if backend_cls.storage_info["device"] == "gpu":
            assert descriptor.storage == dace.StorageType.GPU_Global
        else:
            assert descriptor.storage == dace.StorageType.CPU_Heap

    @pytest.mark.parametrize(
        "backend",
        [
            "dace:cpu",
            pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu]),
        ],
    )
    def test_start_offset(self, backend):
        backend_cls = gt_backend.from_name(backend)
        default_origin = (1, 2, 3)
        stor = gt_store.ones(
            backend=backend,
            managed_memory=False,
            shape=(3, 7, 13),
            default_origin=default_origin,
            dtype=np.float64,
        )
        descriptor: dace.data.Array = stor.__descriptor__()
        raveled = TestDescriptor.ravel_with_padding(stor)[descriptor.start_offset :]
        if backend_cls.storage_info["device"] == "gpu":
            assert raveled.data.ptr % (backend_cls.storage_info["alignment"] * stor.itemsize) == 0
            assert (
                backend_cls.storage_info["alignment"] == 1
                or cp.asarray(stor).data.ptr
                % (backend_cls.storage_info["alignment"] * stor.itemsize)
                != 0
            )
        else:
            assert (
                raveled.ctypes.data % (backend_cls.storage_info["alignment"] * stor.itemsize) == 0
            )
            assert (
                backend_cls.storage_info["alignment"] == 1
                or stor.ctypes.data % (backend_cls.storage_info["alignment"] * stor.itemsize) != 0
            )
