# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import collections
import itertools
import random

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest


try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py
import gt4py.backend as gt_backend
import gt4py.ir as gt_ir
import gt4py.storage as gt_store
import gt4py.storage.utils as gt_storage_utils
import gt4py.utils as gt_utils

from ..definitions import GPU_BACKENDS, INTERNAL_BACKENDS


INTERNAL_GPU_BACKENDS = list(set(GPU_BACKENDS) & set(INTERNAL_BACKENDS))


def generate_data_and_device_data():
    def make_pytest_param(*, validator, test_label, requires_gpu=False, **params):
        test_label = test_label + ", " + ("copy" if params.get("copy", True) else "no copy")
        if requires_gpu:
            marks = [
                pytest.mark.requires_gpu,
                pytest.mark.skipif(cp is None, reason="CuPy not imported."),
            ]
            return pytest.param(params, validator, marks=marks, id=test_label)
        else:
            return pytest.param(params, validator, id=test_label)

    for copy in [True, False]:
        # data np
        test_label = "data np"
        data = np.ones((3, 3))

        def validate(params, data=data, device_data=None, copy=copy):
            assert params["device"] == "cpu"
            assert params["managed"] is False
            np.testing.assert_equal(params["data"], data)
            assert params["data"].ctypes.data == data.ctypes.data
            assert params["device_data"] is None
            assert params["copy"] == copy

        yield make_pytest_param(
            data=data, copy=copy, validator=validate, requires_gpu=False, test_label=test_label
        )

        # data np, device_data cp
        test_label = "data np, device_data cp"
        data = np.ones((3, 3))
        device_data = (cp or np).ones((3, 3))

        def validate(params, data=data, device_data=device_data, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] is False
            np.testing.assert_equal(params["data"], data)
            assert params["data"].ctypes.data == data.ctypes.data
            assert cp.all(params["device_data"] == device_data)
            assert params["device_data"].data.ptr == device_data.data.ptr
            assert params["copy"] == copy

        yield make_pytest_param(
            data=data,
            device_data=device_data,
            copy=copy,
            validator=validate,
            requires_gpu=True,
            test_label=test_label,
        )

        # data cp
        test_label = "data cp"
        data = (cp or np).ones((3, 3))

        def validate(params, data=data, device_data=None, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] is False
            assert params["data"] is None
            assert cp.all(params["device_data"] == data)
            assert params["device_data"].data.ptr == data.data.ptr
            assert params["copy"] == copy

        yield make_pytest_param(
            data=data, copy=copy, validator=validate, requires_gpu=True, test_label=test_label
        )

        # device_data cp
        test_label = "device_data cp"
        device_data = (cp or np).ones((3, 3))

        def validate(params, data=data, device_data=device_data, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] is False
            assert params["data"] is None
            assert cp.all(params["device_data"] == device_data)
            assert params["device_data"].data.ptr == device_data.data.ptr
            assert params["copy"] == copy

        yield make_pytest_param(
            device_data=device_data,
            copy=copy,
            validator=validate,
            requires_gpu=True,
            test_label=test_label,
        )

        # data cp, cuda managed,
        test_label = "data cp, cuda managed"
        data = get_cp_empty_managed((3, 3))

        def validate(params, data=data, device_data=None, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] == "cuda"

            assert isinstance(params["data"], np.ndarray)
            assert params["data"].ctypes.data == data.data.ptr

            assert isinstance(params["device_data"], cp.ndarray)
            assert params["device_data"].data.ptr == data.data.ptr

            assert params["copy"] == copy

        yield make_pytest_param(
            data=data,
            copy=copy,
            validator=validate,
            requires_gpu=True,
            test_label=test_label,
        )
        # data np, cuda managed,
        test_label = "data np, cuda managed"
        data = gt4py.storage.utils._cpu_view(get_cp_empty_managed((3, 3))) if cp else None

        def validate(params, data=data, device_data=None, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] == "cuda"

            assert isinstance(params["data"], np.ndarray)
            assert params["data"].ctypes.data == data.ctypes.data

            assert isinstance(params["device_data"], cp.ndarray)
            assert params["device_data"].data.ptr == data.ctypes.data

            assert params["copy"] == copy

        yield make_pytest_param(
            data=data, copy=copy, validator=validate, requires_gpu=True, test_label=test_label
        )

        # device_data cp, cuda managed,
        test_label = "device_data cp, cuda managed"
        device_data = get_cp_empty_managed((3, 3))

        def validate(params, data=None, device_data=device_data, copy=copy):
            assert params["device"] == "gpu"
            assert params["managed"] == "cuda"

            assert isinstance(params["data"], np.ndarray)
            assert params["data"].ctypes.data == device_data.data.ptr

            assert isinstance(params["device_data"], cp.ndarray)
            assert params["device_data"].data.ptr == device_data.data.ptr

            assert params["copy"] == copy

        yield make_pytest_param(
            device_data=device_data,
            copy=copy,
            validator=validate,
            requires_gpu=True,
            test_label=test_label,
        )

        # data CPU Storage,
        test_label = "data CPU Storage"
        try:
            data = gt_store.empty((3, 3))
        except Exception:
            yield pytest.param(
                None, None, marks=pytest.mark.skip("Failed to construct test case."), id=test_label
            )
        else:
            assert isinstance(data, gt_store.definitions.CPUStorage)

            def validate(params, data=data, device_data=None, copy=copy):
                assert params["device"] == "cpu"
                assert params["managed"] is False
                assert params["device_data"] is None
                assert isinstance(params["data"], np.ndarray)
                assert np.all(params["data"] == np.asarray(data))
                assert params["data"].ctypes.data == data._field.ctypes.data
                assert params["copy"] == copy

            yield make_pytest_param(
                data=data, copy=copy, validator=validate, requires_gpu=False, test_label=test_label
            )

        # data GPU-only Storage,
        test_label = "data GPU-only Storage"
        try:
            data = gt_store.empty((3, 3), device="gpu", managed=False)

        except Exception:
            yield pytest.param(
                None, None, marks=pytest.mark.skip("Failed to construct test case."), id=test_label
            )
        else:
            assert isinstance(data, gt_store.definitions.GPUStorage)

            def validate(params, data=data, device_data=None, copy=copy):
                assert params["device"] == "gpu"
                assert params["managed"] is False
                assert params["data"] is None
                assert isinstance(params["device_data"], cp.ndarray)
                assert cp.all(params["device_data"] == cp.asarray(data))
                assert params["device_data"].data.ptr == data._device_field.data.ptr
                assert params["copy"] == copy

            yield make_pytest_param(
                data=data, copy=copy, validator=validate, requires_gpu=False, test_label=test_label
            )

        # device_data GPU-only Storage,
        test_label = "device_data GPU-only Storage"
        try:
            device_data = gt_store.empty((3, 3), device="gpu", managed=False)
        except Exception:
            yield pytest.param(
                None, None, marks=pytest.mark.skip("Failed to construct test case."), id=test_label
            )
        else:
            assert isinstance(device_data, gt_store.definitions.GPUStorage)

            def validate(params, data=None, device_data=device_data, copy=copy):
                assert params["device"] == "gpu"
                assert params["managed"] is False
                assert params["data"] is None
                assert isinstance(params["device_data"], cp.ndarray)
                assert cp.all(params["device_data"] == cp.asarray(device_data))
                assert params["device_data"].data.ptr == device_data._device_field.data.ptr
                assert params["copy"] == copy

            yield make_pytest_param(
                device_data=device_data,
                copy=copy,
                validator=validate,
                requires_gpu=False,
                test_label=test_label,
            )

        # data Storage, gt4py-managed
        test_label = "data Storage, gt4py-managed"
        try:
            data = gt_store.empty((3, 3), device="gpu", managed="gt4py")
        except Exception:
            yield pytest.param(
                None, None, marks=pytest.mark.skip("Failed to construct test case."), id=test_label
            )
        else:
            assert isinstance(device_data, gt_store.definitions.GPUStorage)

            def validate(params, data=data, device_data=None, copy=copy):
                assert params["device"] == "gpu"
                assert params["managed"] == "gt4py"

                assert isinstance(params["data"], np.ndarray)
                assert params["data"].ctypes.data == data._field.ctypes.data

                assert isinstance(params["device_data"], cp.ndarray)
                assert params["device_data"].data.ptr == data._device_field.data.ptr

                assert params["copy"] == copy
                if copy:
                    assert params["sync_state"] is not data.sync_state
                    assert params["sync_state"].state == data.sync_state.state
                else:
                    assert params["sync_state"] is data.sync_state

            yield make_pytest_param(
                data=data, copy=copy, validator=validate, requires_gpu=False, test_label=test_label
            )

        # data Storage, cuda managed
        test_label = "data Storage, cuda managed"
        try:
            data = gt_store.empty((3, 3), device="gpu", managed="cuda")
        except Exception:
            yield pytest.param(
                None, None, marks=pytest.mark.skip("Failed to construct test case."), id=test_label
            )
        else:
            assert isinstance(data, gt4py.storage.definitions.CudaManagedGPUStorage)

            def validate(params, data=data, device_data=None, copy=copy):
                assert params["device"] == "gpu"
                assert params["managed"] == "cuda"

                assert isinstance(params["data"], np.ndarray)
                assert params["data"].ctypes.data == data._field.ctypes.data

                assert isinstance(params["device_data"], cp.ndarray)
                assert params["device_data"].data.ptr == data._field.ctypes.data

                assert params["copy"] == copy
                assert params["sync_state"] is None

            yield make_pytest_param(
                data=data, copy=copy, validator=validate, requires_gpu=False, test_label=test_label
            )

    reason = "To Implement: tests with __gt_array_interface__"
    yield pytest.param(None, None, marks=pytest.mark.skip(reason), id=reason)


# ---- Hypothesis strategies ----
@hyp_st.composite
def allocation_strategy(draw):
    dtype = np.dtype(
        draw(
            hyp_st.one_of(
                list(hyp_st.just(d) for d in gt4py.ir.nodes.DataType.NUMPY_TO_NATIVE_TYPE)
            )
        )
    )
    alignment_size = (
        draw(hyp_st.one_of(hyp_st.just(1), hyp_st.just(17), hyp_st.just(32))) * dtype.itemsize
    )
    ndim = draw(hyp_st.integers(min_value=0, max_value=5))

    shape_strats = [hyp_st.integers(min_value=0, max_value=10) for _ in range(ndim)]
    shape = draw(hyp_st.tuples(*shape_strats))

    aligned_index_strats = []
    for i in range(ndim):
        aligned_index_strats = aligned_index_strats + [
            hyp_st.integers(min_value=0, max_value=max(0, min(5, shape[i] - 1)))
        ]
    aligned_index = draw(hyp_st.tuples(*aligned_index_strats))
    layout = draw(hyp_st.permutations(tuple(range(ndim))))
    return dict(
        dtype=dtype,
        alignment_size=alignment_size,
        shape=shape,
        aligned_index=aligned_index,
        layout=layout,
    )


@pytest.fixture
def cp_managed_fixture():
    allocator = cp.cuda.get_allocator()
    cp.cuda.set_allocator(cp.cuda.malloc_managed)
    yield
    cp.cuda.set_allocator(allocator)


def get_cp_empty_managed(*args, **kwargs):
    if cp is None:
        return
    allocator = cp.cuda.get_allocator()
    cp.cuda.set_allocator(cp.cuda.malloc_managed)
    try:
        res = cp.empty(*args, **kwargs)
    finally:
        cp.cuda.set_allocator(allocator)
    return res


@pytest.mark.requires_gpu
@pytest.mark.parametrize("shape", [(), (0, 1), (3, 3, 3)])
def test_managed_viewcasting(shape, cp_managed_fixture):

    gpu_arr = cp.empty(shape)
    cpu_view = gt_storage_utils._cpu_view(gpu_arr)
    gpu_view = gt_storage_utils._gpu_view(cpu_view)

    assert cpu_view is not gpu_arr
    assert cpu_view is not gpu_view
    assert gpu_view is not gpu_arr

    assert isinstance(gpu_arr, cp.ndarray)
    assert isinstance(cpu_view, np.ndarray)
    assert isinstance(gpu_view, cp.ndarray)

    if 0 not in shape:
        assert gt_storage_utils.get_ptr(cpu_view) == gt_storage_utils.get_ptr(gpu_arr)
    assert cpu_view.strides == gpu_arr.strides
    assert cpu_view.shape == gpu_arr.shape
    assert cpu_view.dtype == gpu_arr.dtype

    if 0 not in shape:
        assert gt_storage_utils.get_ptr(gpu_view) == gt_storage_utils.get_ptr(gpu_arr)
    assert gpu_view.strides == gpu_arr.strides
    assert gpu_view.shape == gpu_arr.shape
    assert gpu_view.dtype == gpu_arr.dtype

    cpu_view[...] = 123.0
    assert cp.all(gpu_arr == 123.0)
    assert cp.all(gpu_view == 123.0)

    gpu_arr[...] = 321.0
    cp.cuda.Device(0).synchronize()
    assert np.all(cpu_view == 321.0)
    assert cp.all(gpu_view == 321.0)

    gpu_view[...] = 123.0
    cp.cuda.Device(0).synchronize()
    assert np.all(cpu_view == 123.0)
    assert cp.all(gpu_arr == 123.0)


# ---- Tests ----
class TestLowLevelAllocationRoutines:
    def run_test_allocate(self, param_dict, allocate_function):
        aligned_index = param_dict["aligned_index"]
        alignment_size = param_dict["alignment_size"]
        dtype = param_dict["dtype"]
        layout = param_dict["layout"]
        shape = param_dict["shape"]
        ndim = len(shape)

        buffers = allocate_function(aligned_index, shape, layout, dtype, alignment_size)

        # in case of gt4py managed, there are two pairs
        for raw_buffer, field in zip(buffers[::2], buffers[1::2]):
            # check that memory of field is contained in raw_buffer

            if 0 not in shape:
                assert gt_storage_utils.get_ptr(field) >= gt_storage_utils.get_ptr(raw_buffer)
            assert (
                0 in shape
                or ndim == 0
                or gt_storage_utils.get_ptr(field[-1:]) <= gt_storage_utils.get_ptr(raw_buffer[-1:])
            )

            # check if the first compute-domain point in the last dimension is aligned for 100
            # random "columns"

            if ndim > 0 and all(s > 0 for s in shape):
                for _ in range(100):
                    slices = []
                    for hidx in range(ndim):
                        if hidx == np.argmax(layout):
                            slices = slices + [slice(aligned_index[hidx], None)]
                        else:
                            slices = slices + [slice(random.randint(0, shape[hidx]), None)]
                    assert gt_storage_utils.get_ptr(field[tuple(slices)]) % alignment_size == 0

            # check that writing does not give errors, e.g. because of going out of bounds
            if ndim > 0 and all(s > 0 for s in shape):
                slices = []
                for _ in range(ndim):
                    slices = slices + [0]
                field[tuple(slices)] = 1

                slices = []
                for hidx in range(ndim):
                    slices = slices + [aligned_index[hidx]]
                field[tuple(slices)] = 1

                slices = []
                for hidx in range(ndim):
                    slices = slices + [shape[hidx] - 1]
                field[tuple(slices)] = 1

                slices = []
                for hidx in range(ndim):
                    slices = slices + [shape[hidx]]
                with pytest.raises(IndexError):
                    field[tuple(slices)] = 1

            # check if shape is properly set
            assert field.shape == shape
            assert field.dtype == np.dtype(dtype)

        return buffers

    @hyp.given(param_dict=allocation_strategy())
    def test_allocate_cpu(self, param_dict):
        raw_buffer, field = self.run_test_allocate(
            param_dict, allocate_function=gt_storage_utils.allocate_cpu
        )
        assert isinstance(raw_buffer, np.ndarray)
        assert isinstance(field, np.ndarray)

    @pytest.mark.requires_gpu
    @hyp.given(param_dict=allocation_strategy())
    def test_allocate_gpu_only(self, param_dict):
        raw_buffer, field = self.run_test_allocate(
            param_dict, allocate_function=gt_storage_utils.allocate_gpu_only
        )
        assert isinstance(raw_buffer, cp.ndarray)
        assert isinstance(field, cp.ndarray)

    @pytest.mark.requires_gpu
    @hyp.given(param_dict=allocation_strategy())
    def test_allocate_cuda_managed(self, param_dict):
        raw_buffer, field = self.run_test_allocate(
            param_dict, allocate_function=gt_storage_utils.allocate_gpu_cuda_managed
        )
        assert isinstance(raw_buffer, cp.ndarray)
        assert isinstance(field, np.ndarray)

    @pytest.mark.requires_gpu
    @hyp.given(param_dict=allocation_strategy())
    def test_allocate_gt4py_managed(self, param_dict):
        raw_buffer, field, raw_device_buffer, device_field = self.run_test_allocate(
            param_dict, allocate_function=gt_storage_utils.allocate_gpu_gt4py_managed
        )
        assert isinstance(raw_buffer, np.ndarray)
        assert isinstance(field, np.ndarray)
        assert isinstance(raw_device_buffer, cp.ndarray)
        assert isinstance(device_field, cp.ndarray)


class TestNormalizeHalo:
    @pytest.mark.parametrize(
        ["halo_in", "halo_ref"],
        [
            ([1, 2, 3], ((1, 1), (2, 2), (3, 3))),
            ([1, (2, 2), 3], ((1, 1), (2, 2), (3, 3))),
            (None, None),
            ([], ()),
            ((1,), ((1, 1),)),
        ],
    )
    def test_normalize_halo(self, halo_in, halo_ref):
        halo_out = gt_storage_utils.normalize_halo(halo_in)
        assert halo_out is None or gt_utils.is_iterable_of(
            halo_out, iterable_class=tuple, item_class=tuple
        )
        assert halo_out == halo_ref

    @pytest.mark.parametrize(
        ["halo_in", "exc_type"], [("1", TypeError), (1, TypeError), ((-1,), ValueError)]
    )
    def test_normalize_halo_raises(self, halo_in, exc_type):
        # test that exceptions are raised for invalid inputs.
        with pytest.raises(exc_type):
            gt_storage_utils.normalize_halo(halo_in)


def test_normalize_shape():
    from gt4py.storage.utils import normalize_shape

    assert normalize_shape(None) is None
    assert gt_utils.is_iterable_of(normalize_shape([1, 2, 3]), iterable_class=tuple, item_class=int)

    with pytest.raises(TypeError):
        normalize_shape("1")

    with pytest.raises(TypeError):
        normalize_shape(1)

    with pytest.raises(ValueError):
        normalize_shape((0,))


@pytest.mark.parametrize(
    ["device", "managed"],
    [
        ("cpu", False),
        pytest.param("gpu", False, marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "cuda", marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "gt4py", marks=[pytest.mark.requires_gpu]),
    ],
)
def test_slices(device, managed):
    halo = (1, 1, 1)
    shape = (10, 10, 10)
    xp = np if device == "cpu" else cp
    array = xp.random.randn(*shape)
    stor = gt_store.storage(array, device=device, managed=managed, halo=halo)
    sliced = stor[::2, ::2, ::2]
    assert (xp.asarray(sliced) == array[::2, ::2, ::2]).all()
    sliced[...] = array[::2, ::2, ::2]


@pytest.mark.parametrize(
    ["device", "managed"],
    [
        ("cpu", False),
        pytest.param("gpu", False, marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "cuda", marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "gt4py", marks=[pytest.mark.requires_gpu]),
    ],
)
@pytest.mark.parametrize("layout", [(0, 1, 2), (2, 1, 0), (2, 0, 1)])
def test_transpose(device, managed, layout):
    print(layout)
    halo = (1, 1, 1)
    shape = (10, 10, 10)
    array = np.random.randn(*shape)
    stor = gt_store.storage(array, device=device, managed=managed, layout=layout, halo=halo)

    xp = cp if stor.device == "gpu" else np
    ndarray_view = xp.asarray(stor)

    transposed_stor = xp.transpose(stor, axes=(0, 1, 2))
    assert transposed_stor.strides == stor.strides
    assert transposed_stor.shape == stor.shape

    transposed_stor = xp.transpose(stor, axes=(2, 1, 0))
    transposed_np = xp.transpose(ndarray_view, axes=(2, 1, 0))
    assert transposed_stor.strides == transposed_np.strides
    assert transposed_stor.shape == transposed_np.shape

    transposed_np = ndarray_view.transpose(2, 1, 0)
    transposed_stor = stor.transpose(2, 1, 0)
    assert transposed_stor.strides == transposed_np.strides
    assert transposed_stor.shape == transposed_np.shape

    transposed_np = ndarray_view.transpose((2, 1, 0))
    transposed_stor = stor.transpose((2, 1, 0))
    assert transposed_stor.strides == transposed_np.strides
    assert transposed_stor.shape == transposed_np.shape


@pytest.mark.parametrize("method", ["deepcopy", "copy_method"])
class TestCopy:
    @staticmethod
    def assert_not_same_memory_field(stor, stor_copy):
        endslice = tuple(slice(-1, None, None) for s in stor.shape)

        assert stor is not stor_copy
        assert stor._field.ctypes.data != stor_copy._field.ctypes.data
        if stor._field.ctypes.data < stor_copy._field.ctypes.data:
            assert stor._field[endslice].ctypes.data < stor_copy._field.ctypes.data
        else:
            assert stor_copy._field[endslice].ctypes.data < stor._field.ctypes.data

    @staticmethod
    def assert_not_same_memory_device_field(stor, stor_copy):
        endslice = tuple(slice(-1, None, None) for s in stor.shape)
        assert stor is not stor_copy
        assert stor._device_field.data.ptr != stor_copy._device_field.data.ptr
        if stor._device_field.data.ptr < stor_copy._device_field.data.ptr:
            assert stor._device_field[endslice].data.ptr < stor_copy._device_field.data.ptr
        else:
            assert stor_copy._device_field[endslice].data.ptr < stor._device_field.data.ptr

    def test_copy_cpu(self, method):
        halo = (1, 1, 1)
        shape = (10, 10, 10)
        stor = gt_store.storage(np.random.randn(*shape), halo=halo, device="cpu")
        import copy

        if method == "deepcopy":
            stor_copy = copy.deepcopy(stor)
        elif method == "copy_method":
            stor_copy = stor.copy()
        else:
            raise ValueError(f"Test not implemented for copying using '{method}'")

        self.assert_not_same_memory_field(stor, stor_copy)
        np.testing.assert_equal(np.asarray(stor_copy), np.asarray(stor))

    @pytest.mark.requires_gpu
    def test_copy_gpu_only(self, method):
        halo = (1, 1, 1)
        shape = (10, 10, 10)
        stor = gt_store.storage(
            np.random.randn(*shape),
            halo=halo,
            device="gpu",
            managed=False,
        )

        import copy

        if method == "deepcopy":
            stor_copy = copy.deepcopy(stor)
        elif method == "copy_method":
            stor_copy = stor.copy()
        else:
            raise ValueError(f"Test not implemented for copying using '{method}'")

        self.assert_not_same_memory_device_field(stor, stor_copy)
        cp.testing.assert_array_equal(cp.asarray(stor_copy), cp.asarray(stor))

    @pytest.mark.requires_gpu
    def test_copy_gt4py_managed(self, method):
        halo = (1, 1)
        shape = (3, 3)
        data = np.random.randn(*shape)
        stor = gt_store.storage(
            data,
            halo=halo,
            device="gpu",
            managed="gt4py",
        )

        import copy

        if method == "deepcopy":
            stor_copy = copy.deepcopy(stor)
        elif method == "copy_method":
            stor_copy = stor.copy()
        else:
            raise ValueError(f"Test not implemented for copying using '{method}'")
        self.assert_not_same_memory_field(stor, stor_copy)
        self.assert_not_same_memory_device_field(stor, stor_copy)

        assert stor.sync_state is not stor_copy.sync_state

        stor.device_to_host()
        stor_copy.device_to_host()
        np.testing.assert_equal(np.asarray(stor_copy), np.asarray(stor))

        stor.host_to_device()
        stor_copy.host_to_device()
        cp.testing.assert_array_equal(cp.asarray(stor_copy), cp.asarray(stor))

    @pytest.mark.requires_gpu
    def test_copy_cuda_managed(self, method):
        halo = (1, 1, 1)
        shape = (10, 10, 10)
        stor = gt_store.storage(np.random.randn(*shape), halo=halo, device="gpu", managed="cuda")

        import copy

        if method == "deepcopy":
            stor_copy = copy.deepcopy(stor)
        elif method == "copy_method":
            stor_copy = stor.copy()
        else:
            raise ValueError(f"Test not implemented for copying using '{method}'")

        self.assert_not_same_memory_field(stor, stor_copy)

        np.testing.assert_equal(np.asarray(stor_copy), np.asarray(stor))


@pytest.mark.requires_gpu
@pytest.mark.parametrize("managed", [False, "cuda", "gt4py"])
def test_cuda_array_interface(managed):
    data = cp.random.randn(5, 5, 5)
    storage = gt_store.storage(data, device="gpu", managed=managed)
    cupy_array = cp.array(storage)
    assert (cupy_array == data).all()


class TestParameterLookupAndNormalizeValid:
    """Tests storage construction parameter lookup routine.

    The GDP-3 (Duck Storages) states:
        The values of parameters which are not explicitly defined by the user will be inferred from
        the first alternative source where the parameter is defined in the following search order:

        1. The provided :code:`defaults` parameter set.
        2. The provided :code:`data` or :code:`device_data` parameters.
        3. A fallback default value specified above. The only case where this is not available is
           :code:`shape`, in which case an exception is raised.

    The tests in this class should test that this lookup order is implemented correctly
    by the `parameter_lookup_and_normalize` utility for all parameters.

    string parameters of pattern "ref:key" indicate that the result is the same instance as the
    key "key" in input_params

    Tests with the name according to the pattern test_normalize_* check that the resulting value
    is of the right type and canonical form

    Tests with the name according to the pattern test_lookup* check that the resulting value
    is right according to the lookup order (i.e. the order specified at the beginning of this
    docstring).
    """

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": [1, np.int32(2), 3]}, {"shape": (1, 2, 3)}),
            ({"copy": True, "data": 3.0, "shape": (1, 2, 3)}, {"shape": (1, 2, 3), "data": 3.0}),
            ({"copy": True, "data": np.ones((1, 2, 3))}, {"shape": (1, 2, 3), "data": "ref:data"}),
            ({"copy": False, "template": np.ones((1, 2, 3))}, {"shape": (1, 2, 3), "data": None}),
        ],
    )
    def test_lookup_shape(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [({"copy": False, "shape": [1, np.int32(2), 3]}, {"shape": (1, 2, 3)})],
    )
    def test_normalize_shape(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": (1, 2, 3)}, {"dtype": np.dtype("float64")}),
            (
                {"copy": False, "shape": (1, 2, 3), "dtype": np.dtype("int64")},
                {"dtype": np.dtype("int64")},
            ),
            (
                {"copy": False, "data": np.ones((1, 2, 3), dtype=np.float32)},
                {"dtype": np.dtype("float32")},
            ),
            (
                {
                    "copy": True,
                    "data": np.ones((1, 2, 3), dtype=np.float32),
                    "template": np.ones((1, 2, 3), dtype=np.int32),
                },
                {"dtype": np.dtype("int32")},
            ),
            (
                {"copy": False, "template": np.ones((1, 2, 3), dtype=np.int32)},
                {"dtype": np.dtype("int32")},
            ),
        ],
    )
    def test_lookup_dtype(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": (1, 2, 3), "dtype": float}, {"dtype": np.dtype("float64")}),
            ({"copy": False, "shape": (1, 2, 3), "dtype": np.int32}, {"dtype": np.dtype("int32")}),
            ({"copy": False, "shape": (1, 2, 3), "dtype": "b1"}, {"dtype": np.dtype("bool")}),
            (
                {"copy": False, "shape": (1, 2, 3), "dtype": gt_ir.DataType.FLOAT32},
                {"dtype": np.dtype("float32")},
            ),
            pytest.param(
                {"copy": False, "shape": (1, 2, 3), "dtype": (cp or np).float32},
                {"dtype": np.dtype("float32")},
                marks=[
                    pytest.mark.requires_gpu,
                    pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                ],
            ),
        ],
    )
    def test_normalize_dtype(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            # test parameter group: aligned index and halo
            # test default halo
            (
                {"copy": False, "shape": (4, 4, 4, 3), "aligned_index": (2, 2, 2, 1)},
                {"aligned_index": (2, 2, 2, 1), "halo": ((0, 0), (0, 0), (0, 0), (0, 0))},
            ),
            # test default aligned_index
            (
                {"copy": False, "shape": (4, 4), "halo": ((2, 1), (2, 2))},
                {"aligned_index": (2, 2), "halo": ((2, 1), (2, 2))},
            ),
            # test if both aligned_index and halo are explicitly set, those values are used.
            (
                {
                    "copy": False,
                    "shape": (4, 4),
                    "halo": ((2, 1), (0, 2)),
                    "aligned_index": (0, 1),
                },
                {"aligned_index": (0, 1), "halo": ((2, 1), (0, 2))},
            ),
            # test defaults if neither aligned_index nor halo are set
            (
                {"copy": False, "shape": (4, 4)},
                {"aligned_index": (0, 0), "halo": ((0, 0), (0, 0))},
            ),
        ],
    )
    def test_lookup_aligned_index_and_halo(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            (
                {"copy": False, "shape": (4, 4), "aligned_index": [2, np.int32(2)]},
                {"aligned_index": (2, 2)},
            )
        ],
    )
    def test_normalize_aligned_index(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": (4, 4), "alignment_size": 7}, {"alignment_size": 7}),
            (
                {"copy": False, "shape": (4, 4), "defaults": "gtmc", "alignment_size": 7},
                {"alignment_size": 7},
            ),
            ({"copy": False, "shape": (4, 4), "defaults": "gtmc"}, {"alignment_size": 8}),
        ],
    )
    def test_normalize_alignment_size(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": (4, 4, 5), "layout": (0, 1, 2)}, {"layout": (0, 1, 2)}),
            (
                {"copy": False, "shape": (4, 5, 7), "defaults": "F", "dims": "IJK"},
                {"layout": (2, 1, 0)},
            ),
            (
                {
                    "copy": False,
                    "shape": (4, 4, 5, 7),
                    "defaults": "F",
                    "dims": ["10", "J", "K", "I"],
                },
                {"layout": (3, 2, 1, 0)},
            ),
            (
                {"copy": False, "shape": (4, 4, 5, 7), "defaults": "gtcuda", "dims": "IJK0"},
                {"layout": (3, 2, 1, 0)},
            ),
            (
                {
                    "copy": False,
                    "shape": (4, 4, 5, 7),
                    "defaults": "gtcuda",
                    "dims": ["10", "J", "K", "I"],
                },
                {"layout": (0, 2, 1, 3)},
            ),
            (
                {"copy": False, "shape": (4, 4, 5), "layout": lambda dims: (0, 1, 2)},
                {"layout": (0, 1, 2)},
            ),
        ],
    )
    def test_lookup_layout(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            (
                {"copy": False, "shape": (4, 4, 5), "layout": [0, np.int32(1), 2]},
                {"layout": (0, 1, 2)},
            ),
            (
                {"copy": False, "shape": (4, 4, 5), "layout": lambda ndim: [0, np.int32(1), 2]},
                {"layout": (0, 1, 2)},
            ),
            ({"copy": False, "shape": (4, 4, 5), "defaults": "gtx86"}, {"layout": (0, 1, 2)}),
        ],
    )
    def test_normalize_layout(self, input_params, resolved_params):  # callable x tuple > tuple
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(["input_params", "validator"], iter(generate_data_and_device_data()))
    def test_lookup_data_device_data(self, input_params, validator):
        res_params = self.run_lookup_normalize(input_params)
        validator(res_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            (
                {"copy": False, "data": np.ones((3, 3, 3))},
                {"managed": False},
            ),
            pytest.param(
                {"copy": False, "data": (cp or np).ones((3, 3, 3))},
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": False, "data": get_cp_empty_managed((3, 3, 3))},
                {"managed": "cuda"},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": False, "managed": False, "data": (cp or np).empty((3, 3, 3))},
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": False, "managed": "cuda", "data": get_cp_empty_managed((3, 3, 3))},
                {"managed": "cuda"},
                marks=[pytest.mark.requires_gpu],
            ),
            ({"copy": True, "managed": False, "data": np.empty((3, 3, 3))}, {"managed": False}),
            pytest.param(
                {"copy": True, "managed": False, "data": (cp or np).empty((3, 3, 3))},
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": True, "managed": False, "data": get_cp_empty_managed((3, 3, 3))},
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            ({"copy": True, "managed": "cuda", "data": np.empty((3, 3, 3))}, {"managed": "cuda"}),
            pytest.param(
                {"copy": True, "managed": "cuda", "data": (cp or np).empty((3, 3, 3))},
                {"managed": "cuda"},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": True, "managed": "cuda", "data": get_cp_empty_managed((3, 3, 3))},
                {"managed": "cuda"},
                marks=[pytest.mark.requires_gpu],
            ),
            (
                {"copy": True, "managed": "gt4py", "data": np.empty((3, 3, 3))},
                {"managed": "gt4py"},
            ),
            pytest.param(
                {"copy": True, "managed": "gt4py", "data": (cp or np).empty((3, 3, 3))},
                {"managed": "gt4py"},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {"copy": True, "managed": "gt4py", "data": get_cp_empty_managed((3, 3, 3))},
                {"managed": "gt4py"},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {
                    "copy": False,
                    "data": gt_store.empty((3, 3, 3), device="cpu", managed=False) if cp else None,
                },
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {
                    "copy": False,
                    "data": gt_store.empty((3, 3, 3), device="gpu", managed=False) if cp else None,
                },
                {"managed": False},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {
                    "copy": False,
                    "data": gt_store.empty((3, 3, 3), device="gpu", managed="gt4py")
                    if cp
                    else None,
                },
                {"managed": "gt4py"},
                marks=[pytest.mark.requires_gpu],
            ),
            pytest.param(
                {
                    "copy": False,
                    "data": gt_store.empty((3, 3, 3), device="gpu", managed="cuda") if cp else None,
                },
                {"managed": "cuda"},
                marks=[pytest.mark.requires_gpu],
            ),
        ],
    )
    def test_lookup_managed(self, input_params, resolved_params):
        self.run_test_lookup_normalize(input_params, resolved_params)

    @pytest.mark.parametrize(
        ["input_params", "resolved_params"],
        [
            ({"copy": False, "shape": (4, 4, 5)}, {"device": "cpu"}),
            ({"copy": False, "shape": (4, 4, 5), "device": "gpu"}, {"device": "gpu"}),
            ({"copy": False, "shape": (4, 4, 5), "defaults": "debug"}, {"device": "cpu"}),
            ({"copy": False, "shape": (4, 4, 5), "defaults": "gtcuda"}, {"device": "gpu"}),
            (
                {"copy": False, "shape": (4, 4, 5), "defaults": "gtcuda", "device": "cpu"},
                {"device": "cpu"},
            ),
            ({"copy": False, "shape": (4, 4, 5), "data": np.ones((4, 4, 5))}, {"device": "cpu"}),
            pytest.param(
                {
                    "copy": False,
                    "shape": (4, 4, 5),
                    "data": (cp or np).ones((4, 4, 5)),
                    "device": "cpu",
                },
                {"device": "cpu"},
                marks=[
                    pytest.mark.requires_gpu,
                    pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                ],
            ),
            pytest.param(
                {"copy": False, "shape": (4, 4, 5), "data": (cp or np).ones((4, 4, 5))},
                {"device": "gpu"},
                marks=[
                    pytest.mark.requires_gpu,
                    pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                ],
            ),
            pytest.param(
                {"copy": False, "shape": (4, 4, 5), "device_data": (cp or np).ones((4, 4, 5))},
                {"device": "gpu"},
                marks=[
                    pytest.mark.requires_gpu,
                    pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                ],
            ),
            *[
                pytest.param(
                    {"copy": False, "shape": (4, 4, 5), "defaults": name},
                    {"device": "gpu"},
                    marks=[
                        pytest.mark.requires_gpu,
                        pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                    ],
                )
                for name in INTERNAL_GPU_BACKENDS
                if gt_backend.from_name(name).compute_device == "gpu"
            ],
            *[
                pytest.param(
                    {"data": np.ones((4, 4, 5)), "copy": False, "defaults": name},
                    {"device": "gpu"},
                    marks=[
                        pytest.mark.requires_gpu,
                        pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                    ],
                )
                for name in INTERNAL_GPU_BACKENDS
                if gt_backend.from_name(name).compute_device == "gpu"
            ],
            *[
                pytest.param(
                    {"template": np.ones((4, 4, 5)), "copy": False, "defaults": name},
                    {"device": "gpu"},
                    marks=[
                        pytest.mark.requires_gpu,
                        pytest.mark.skipif(cp is None, reason="CuPy not imported."),
                    ],
                )
                for name in INTERNAL_GPU_BACKENDS
                if gt_backend.from_name(name).compute_device == "gpu"
            ],
        ],
    )
    def test_lookup_device(
        self, input_params, resolved_params
    ):  # defaults, explicit, data, global default,
        self.run_test_lookup_normalize(input_params, resolved_params)

    def run_lookup_normalize(self, input_params):

        params = {
            "aligned_index": None,  #
            "alignment_size": None,  #
            "data": None,  #
            "defaults": None,  #
            "device": None,
            "device_data": None,  #
            "dtype": None,  #
            "dims": None,
            "halo": None,  # ... if template is storage
            "layout": None,
            "managed": None,
            "shape": None,
            "sync_state": None,
            "template": None,
        }
        use_params = dict(params)
        use_params.update(**input_params)

        res_params = gt_storage_utils.parameter_lookup_and_normalize(**use_params)

        res_keys = [
            "aligned_index",
            "alignment_size",
            "copy",
            "data",
            "device",
            "device_data",
            "dtype",
            "halo",
            "layout",
            "managed",
            "shape",
            "sync_state",
        ]
        assert len(res_params) == len(res_keys)
        for key in res_keys:
            assert key in res_params

        return res_params

    def run_test_lookup_normalize(self, input_params, resolved_params):
        res_params = self.run_lookup_normalize(input_params)

        solution_params = dict(resolved_params)
        for key, value in dict(solution_params).items():
            if isinstance(value, str) and value.startswith("ref:"):
                solution_params[key] = input_params[value[4:]]

        for key, value in solution_params.items():

            assert np.all(res_params[key] == value), f"Wrong value for parameter '{key}'."
            assert type(res_params[key]) is type(value), f"Wrong type for parameter '{key}'."
            if isinstance(res_params[key], collections.Iterable):
                assert all(
                    type(v1) is type(v2) for v1, v2 in zip(res_params[key], value)
                ), f"Wrong type for parameter '{key}'."

            if "ref:" + key in resolved_params:
                assert res_params[key] is input_params[key]


@pytest.mark.parametrize(
    "layout", [pytest.param(s, id=str(s)) for s in itertools.permutations((0, 1, 2))]
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(s, id=str(s))
        for s in [
            (3, 3, 3),
            (3, 3, 1),
            (3, 1, 3),
            (1, 3, 3),
            (3, 1, 1),
            (1, 3, 1),
            (1, 1, 3),
            (1, 1, 1),
        ]
    ],
)
def test_layout_from_strides(layout, shape):
    _, array = gt_storage_utils.allocate_cpu((0, 0, 0), shape, layout, np.float64, 8)
    proposed_layout = gt_storage_utils.layout_from_strides(array.strides)
    assert gt_storage_utils.is_compatible_layout(array.strides, shape, proposed_layout)
