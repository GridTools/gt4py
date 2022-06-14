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
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, np.ndarray)


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
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, cp.ndarray)


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
def test_slice_gpu():
    stor = gt_store.ones(
        backend="gt:gpu",
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
        descriptor: dace.data.Array = gt_store.dace_descriptor(
                backend=backend,
                shape=(3, 7, 13),
                default_origin=(1, 2, 3),
                dtype=np.float64,
            )
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
            shape=(3, 7, 13),
            default_origin=default_origin,
            dtype=np.float64,
        )
        descriptor: dace.data.Array = gt_store.dace_descriptor(
                backend=backend,
                shape=(3, 7, 13),
                default_origin=(1, 2, 3),
                dtype=np.float64,
            )
        raveled = TestDescriptor.ravel_with_padding(stor)[descriptor.start_offset :]
        if backend_cls.storage_info["device"] == "gpu":
            assert raveled.data.ptr % (backend_cls.storage_info["alignment"] * stor.itemsize) == 0
            assert (
                backend_cls.storage_info["alignment"] == 1
                or stor.data.ptr
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
