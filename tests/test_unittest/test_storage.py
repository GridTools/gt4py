# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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


import numpy as np
import math
import random

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import gt4py
import gt4py.storage as gt_storage
from gt4py.storage.utils import *
import gt4py.storage.utils as gt_storage_utils
from gt4py import backend as gt_backend
from gt4py.storage.storage import Storage
import gt4py.utils as gt_utils

import itertools

CPU_STORAGE_KEYS = set(
    name
    for name, parameters in gt4py.storage.default_parameters.REGISTRY.items()
    if not parameters.gpu
)

# ---- Hypothesis strategies ----
@hyp_st.composite
def allocation_strategy(draw):
    dtype = np.dtype(
        draw(hyp_st.one_of(list(hyp_st.just(d) for d in gt4py.ir.nodes.DataType.supported_dtypes)))
    )
    alignment = (
        draw(hyp_st.one_of(hyp_st.just(1), hyp_st.just(17), hyp_st.just(32))) * dtype.itemsize
    )
    ndim = draw(hyp_st.integers(min_value=0, max_value=3))

    shape_strats = []
    default_origin_strats = []

    shape_strats = [hyp_st.integers(min_value=0, max_value=10) for i in range(ndim)]
    shape = draw(hyp_st.tuples(*shape_strats))
    for i in range(ndim):
        default_origin_strats = default_origin_strats + [
            hyp_st.integers(min_value=0, max_value=max(0, min(5, shape[i] - 1)))
        ]
    default_origin = draw(hyp_st.tuples(*default_origin_strats))
    layout_order = draw(hyp_st.permutations(tuple(range(ndim))))
    return dict(
        dtype=dtype,
        alignment=alignment,
        shape=shape,
        default_origin=default_origin,
        layout_order=layout_order,
    )


@hyp_st.composite
def axes_strategy(draw):
    dimension = draw(hyp_st.integers(min_value=0, max_value=3))

    shape = draw(
        hyp_st.tuples(*[hyp_st.integers(min_value=0, max_value=10) for i in range(dimension)])
    )
    default_origin = draw(
        hyp_st.tuples(
            *[
                hyp_st.integers(min_value=0, max_value=max(0, min(5, shape[i] - 1)))
                for i in range(dimension)
            ]
        )
    )

    axes = draw(
        hyp_st.one_of(
            hyp_st.just(None),
            hyp_st.lists(hyp_st.booleans(), min_size=3, max_size=3).filter(
                lambda x: sum(x) == dimension
            ),
        )
    )
    if isinstance(axes, list):
        axes = [a for m, a in zip(axes, "IJK") if m]
    return dict(shape=shape, default_origin=default_origin, axes=axes)


# ---- Tests ----
def test_allocate_cpu():
    run_test_allocate_singledevice(allocate_function=allocate_cpu)


@pytest.mark.requires_gpu
def test_allocate_gpu_only():
    run_test_allocate_singledevice(allocate_function=allocate_gpu_only)


@hyp.given(param_dict=allocation_strategy())
def run_test_allocate_singledevice(param_dict, allocate_function):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]
    ndim = len(shape)

    raw_buffer, field = allocate_function(
        default_origin, shape, layout_map, dtype, alignment_bytes
    )

    # check that memory of field is contained in raw_buffer
    assert get_ptr(field) >= get_ptr(raw_buffer)
    assert ndim == 0 or get_ptr(field[-1:]) <= get_ptr(raw_buffer[-1:])

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    if ndim > 0 and all(s > 0 for s in shape):
        for i in range(100):
            slices = []
            for hidx in range(ndim):
                if hidx == np.argmax(layout_map):
                    slices = slices + [slice(default_origin[hidx], None)]
                else:
                    slices = slices + [slice(random.randint(0, shape[hidx]), None)]
            assert get_ptr(field[tuple(slices)]) % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    if ndim > 0 and all(s > 0 for s in shape):
        slices = []
        for hidx in range(ndim):
            slices = slices + [0]
        field[tuple(slices)] = 1

        slices = []
        for hidx in range(ndim):
            slices = slices + [default_origin[hidx]]
        field[tuple(slices)] = 1

        slices = []
        for hidx in range(ndim):
            slices = slices + [shape[hidx] - 1]
        field[tuple(slices)] = 1

        slices = []
        for hidx in range(ndim):
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
def test_allocate_cuda_managed(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]

    raw_buffer, field = allocate_gpu(default_origin, shape, layout_map, dtype, alignment_bytes)

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
def test_allocate_gpu_gt4py_managed(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    default_origin = param_dict["default_origin"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]
    raw_buffer, field, device_raw_buffer, device_field = allocate_gpu_unmanaged(
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


def test_normalize_default_origin():
    from gt4py.storage.utils import normalize_shape

    assert gt_utils.is_iterable_of(
        normalize_default_origin([1, 2, 3]), iterable_class=tuple, item_class=int
    )

    # test that exceptions are raised for invalid inputs.
    try:
        normalize_default_origin("1")
    except TypeError:
        pass
    else:
        assert False

    try:
        normalize_default_origin(1)
    except TypeError:
        pass
    else:
        assert False

    try:
        normalize_default_origin((-1,))
    except ValueError:
        pass
    else:
        assert False


def test_normalize_shape():
    from gt4py.storage.utils import normalize_shape

    assert normalize_shape(None) is None
    assert gt_utils.is_iterable_of(
        normalize_shape([1, 2, 3]), iterable_class=tuple, item_class=int
    )

    # test that exceptions are raised for invalid inputs.
    try:
        normalize_shape("1")
    except TypeError:
        pass
    else:
        assert False

    try:
        normalize_shape(1)
    except TypeError:
        pass
    else:
        assert False

    try:
        normalize_shape((0,))
    except ValueError:
        pass
    else:
        assert False


@hyp.given(hyp_data=hyp_st.data())
def test_cpustorage_init(hyp_data):

    from gt4py.storage.storage import CPUStorage

    ndim = hyp_data.draw(hyp_st.integers(min_value=1, max_value=3), label="ndim")
    shape = hyp_data.draw(
        hyp_st.tuples(*([hyp_st.integers(min_value=2, max_value=2)] * ndim)), label="shape"
    )
    mask = hyp_data.draw(
        hyp_st.lists(hyp_st.booleans(), min_size=3, max_size=3).filter(lambda x: sum(x) == ndim),
        label="mask",
    )
    alignment = hyp_data.draw(
        hyp_st.one_of([hyp_st.just(a) for a in {1, 13, 32}]), label="alignment"
    )

    supported_typecodes = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
    dtype = hyp_data.draw(
        hyp_st.one_of([hyp_st.just(c) for c in supported_typecodes]), label="dtype"
    )
    layout_map = hyp_data.draw(hyp_st.permutations(range(3)), label="layout_map")

    data = np.array(np.random.randn(*shape)).astype(dtype)

    # without data
    storage = CPUStorage(
        shape=shape,
        default_origin=(0, 0, 0),
        dtype=dtype,
        array=data,
        mask=mask,
        alignment=alignment,
        layout_map=layout_map,
    )

    assert isinstance(storage.shape, tuple)
    assert storage.shape == shape
    assert isinstance(storage._expanded_shape, tuple)

    assert storage.mask == tuple(mask)

    assert all([a in "IJK" for a in storage.axes])
    assert isinstance(storage.axes, str)
    assert len(storage.axes) == ndim
    assert len(set(storage.axes)) == ndim

    assert tuple(es for m, es in zip(storage.mask, storage._expanded_shape) if m) == shape
    assert all([es == 1 for m, es in zip(storage.mask, storage._expanded_shape) if not m])

    assert isinstance(storage.dtype, np.dtype)
    assert storage.dtype is np.dtype(dtype)

    storage = CPUStorage(
        array=data,
        shape=shape,
        default_origin=(0, 0, 0),
        dtype=dtype,
        mask=mask,
        alignment=alignment,
        device=None,
        layout_map=layout_map,
        copy=True,
    )
    if ndim > 0 and np.prod(shape) > 0:
        slices = []
        for d in range(ndim):
            slices.append(slice(-1, None, 1))
        assert storage._ptr > get_ptr(data[slices]) or get_ptr(
            np.asarray(storage)[slices]
        ) < get_ptr(data)
    assert np.all(np.asarray(storage) == data)

    storage = CPUStorage(
        array=data,
        shape=shape,
        default_origin=(0, 0, 0),
        dtype=dtype,
        mask=mask,
        alignment=alignment,
        device=None,
        layout_map=layout_map,
        copy=False,
    )
    assert storage._ptr == get_ptr(data)
    assert np.all(np.asarray(storage) == data)


class TestCPUConstructors:
    @staticmethod
    @pytest.mark.parametrize(
        ["alloc_fun", "default_parameters"],
        itertools.product(
            [
                gt_storage.empty,
                gt_storage.ones,
                gt_storage.zeros,
                lambda dtype, default_origin, shape, default_parameters: gt_storage.full(
                    3,
                    default_parameters=default_parameters,
                    shape=shape,
                    dtype=dtype,
                    default_origin=default_origin,
                ),
            ],
            CPU_STORAGE_KEYS,
        ),
    )
    def test_scalar_constructors(alloc_fun, default_parameters):
        stor = alloc_fun(
            dtype=np.int32,
            default_origin=(1, 2, 3),
            shape=(2, 4, 6),
            default_parameters=default_parameters,
        )
        assert stor.default_origin == (1, 2, 3)
        assert stor.shape == (2, 4, 6)
        assert isinstance(stor, Storage)
        if alloc_fun.__name__ == "ones":
            assert np.all(stor == 1)
        elif alloc_fun.__name__ == "zeros":
            assert np.all(stor == 0)
        elif alloc_fun.__name__ == "<lambda>":
            assert np.all(stor == 3)
        else:
            assert alloc_fun.__name__ == "empty", "ill-defined test"

    @staticmethod
    @pytest.mark.parametrize("default_parameters", CPU_STORAGE_KEYS)
    def test_asstorage(default_parameters):
        data = np.array(np.random.randn(2, 4, 6), dtype=np.float64)
        stor = gt_storage.asstorage(
            data,
            dtype=np.float64,
            default_origin=(1, 2, 3),
            shape=(2, 4, 6),
            default_parameters=default_parameters,
        )
        from gt4py.storage.storage import CPUStorage

        assert isinstance(stor, CPUStorage)

        assert stor.default_origin == (1, 2, 3)
        assert stor.shape == (2, 4, 6)
        assert isinstance(stor, Storage)
        assert np.all(data == np.asarray(stor))
        assert get_ptr(data) == stor._ptr

    @staticmethod
    @pytest.mark.parametrize("default_parameters", CPU_STORAGE_KEYS)
    @pytest.mark.parametrize("method", [Storage, gt_storage.storage])
    def test_storage(method, default_parameters):
        data = np.array(np.random.randn(2, 4, 6), dtype=np.float64)
        stor = method(
            data,
            dtype=np.float64,
            default_origin=(1, 2, 3),
            shape=(2, 4, 6),
            default_parameters=default_parameters,
        )
        assert stor.default_origin == (1, 2, 3)
        assert stor.shape == (2, 4, 6)
        assert isinstance(stor, Storage)
        assert get_ptr(data) != stor._ptr
        slices = []
        for d in range(3):
            slices.append(slice(-1, None, 1))
        assert stor._ptr > get_ptr(data[slices]) or get_ptr(np.asarray(stor)[slices]) < get_ptr(
            data
        )
        assert np.all(data == np.asarray(stor))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["alloc_fun", "backend"],
    itertools.product(
        [
            gt_storage.empty,
            gt_storage.ones,
            gt_storage.zeros,
            lambda dtype, default_origin, shape, backend: gt_storage.from_array(
                np.empty(shape, dtype=dtype),
                backend=backend,
                shape=shape,
                dtype=dtype,
                default_origin=default_origin,
            ),
        ],
        [
            name
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).compute_device == "gpu"
        ],
    ),
)
def test_gpu_constructor(alloc_fun, backend):
    stor = alloc_fun(dtype=np.float64, default_origin=(1, 2, 3), shape=(2, 4, 6), backend=backend)
    assert stor.default_origin == (1, 2, 3)
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, np.ndarray)
    assert stor.is_stencil_view


class TestMaskedStorage:
    @staticmethod
    @hyp.given(param_dict=axes_strategy())
    def test_masked_storage(param_dict):
        axes = param_dict["axes"]
        default_origin = param_dict["default_origin"]
        shape = param_dict["shape"]

        # no assert when all is defined in descriptor, no grid_group
        store = gt_storage.empty(
            dtype=np.float64,
            default_origin=default_origin,
            shape=shape,
            axes=axes,
            default_parameters="gtx86",
        )
        if axes is not None:
            assert store.axes == "".join(axes)
            assert store._mask == tuple(a in axes for a in "IJK")
        assert sum(store.mask) == store.ndim
        assert sum(store.mask) == len(store.shape)

    @staticmethod
    @pytest.mark.parametrize("axes", ["II", "JI", "IJP"])
    def test_masked_storage_assert(axes):
        default_origin = (1, 1, 1)
        shape = (2, 2, 2)

        with pytest.raises(ValueError):
            gt_storage.empty(
                dtype=np.float64,
                default_origin=default_origin,
                shape=shape,
                axes=axes,
                default_parameters="gtx86",
            )


def run_test_slices(backend):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = np.random.randn(*shape)
    stor = gt_storage.storage(
        array,
        default_parameters=backend,
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
    )
    sliced = stor[::2, ::2, ::2]
    assert np.all(sliced == array[::2, ::2, ::2])
    sliced[...] = array[::2, ::2, ::2]


def test_slices_cpu():
    run_test_slices(backend="gtmc")


@pytest.mark.requires_gpu
def test_slices_gpu():
    run_test_slices(backend="gtcuda")

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_storage.from_array(
        array, backend="gtcuda", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    sliced = stor[::2, ::2, ::2]
    # assert (sliced == array[::2, ::2, ::2]).all()
    sliced[...] = array[::2, ::2, ::2]

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_storage.from_array(
        array, backend="gtcuda", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    ref = gt_storage.from_array(
        array, backend="gtcuda", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import cupy as cp
    import copy

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_storage.from_array(
        array, backend="gtcuda", dtype=np.float64, default_origin=default_origin, shape=shape
    )
    ref = copy.deepcopy(stor)
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import cupy as cp

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_storage.from_array(
        array,
        backend="gtcuda",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=True,
    )
    ref = gt_storage.from_array(
        array,
        backend="gtcuda",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=True,
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2]

    import cupy as cp
    import copy

    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = cp.random.randn(*shape)
    stor = gt_storage.from_array(
        array,
        backend="gtcuda",
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
    stor = gt_storage.from_array(
        array,
        backend="gtcuda",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=False,
    )
    ref = gt_storage.from_array(
        array,
        backend="gtcuda",
        dtype=np.float64,
        default_origin=default_origin,
        shape=shape,
        managed_memory=False,
    )
    # assert (sliced == array[::2, ::2, ::2]).all()
    stor[::2, ::2, ::2] = ref[::2, ::2, ::2] + ref[::2, ::2, ::2]


def test_transpose(backend="gtmc"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    array = np.random.randn(*shape)
    stor = gt_storage.storage(
        array, default_origin=default_origin, default_parameters=backend, dtype=np.float64
    )
    transposed = np.transpose(stor, axes=(0, 1, 2))
    assert transposed.strides == stor.strides
    with pytest.raises(NotImplementedError):
        np.transpose(stor, axes=(2, 1, 0))
    with pytest.raises(NotImplementedError):
        np.transpose(stor)


@pytest.mark.parametrize(
    ["backend", "method"],
    itertools.product(
        [
            name
            for name, parameters in gt4py.storage.default_parameters.REGISTRY.items()
            if not parameters.gpu
        ],
        ["deepcopy", "copy_method"],
    ),
)
def test_copy_cpu(method, backend):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_storage.storage(
        np.random.randn(*shape),
        default_origin=default_origin,
        default_parameters=backend,
        dtype=np.float64,
    )

    import copy

    if method == "deepcopy":
        stor_copy = copy.deepcopy(stor)
    elif method == "copy_method":
        stor_copy = stor.copy()
    else:
        raise ValueError(f"Test not implemented for copying using '{method}'")

    assert stor is not stor_copy
    assert stor._ptr != stor_copy._ptr
    slices = []
    assert stor._ptr > stor_copy._ptr or stor[-1:, -1:, -1:]._ptr < stor_copy._ptr
    stor[-1:, -1:, -1:]
    assert np.all(stor == stor_copy)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("method", ["deepcopy", "copy_method"])
def test_copy_gpu(method, backend="gtcuda"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_storage.from_array(
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
def test_deepcopy_gpu_unmanaged(method, backend="gtcuda"):
    default_origin = (1, 1, 1)
    shape = (10, 10, 10)
    stor = gt_storage.from_array(
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


@pytest.mark.requires_gpu
def test_cuda_array_interface():
    storage = gt_storage.from_array(
        cp.random.randn(5, 5, 5),
        backend="gtcuda",
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
    gpu_arr = cp.empty((5, 5, 5))

    gpu_arr = cp.ones((10, 10, 10))
    cpu_view = gt_storage_utils.cpu_view(gpu_arr)

    gpu_arr[0, 0, 0] = 123
    assert cpu_view[0, 0, 0] == 123
    cpu_view[1, 1, 1] = 321
    assert gpu_arr[1, 1, 1] == 321


import operator as op


@hyp.given(data=hyp_st.data())
def test_ufunc_broadcast(data):

    backend = "gtmc"
    dtype = np.float64

    ndims = data.draw(hyp_st.integers(min_value=0, max_value=3), label="ndims")

    mask_st = hyp_st.lists(hyp_st.booleans(), min_size=ndims, max_size=ndims)
    mask1 = data.draw(mask_st, label="mask1")
    axes1 = [a for m, a in zip(mask1, "IJK") if m]
    mask2 = data.draw(mask_st, label="mask2")
    axes2 = [a for m, a in zip(mask2, "IJK") if m]

    semantic_shape = tuple(
        data.draw(
            hyp_st.lists(
                # hyp_st.integers(min_value=0, max_value=5), min_size=ndims, max_size=ndims
                hyp_st.integers(min_value=0, max_value=5),
                min_size=ndims,
                max_size=ndims,
            ),
            label="semantic_shape",
        )
    )

    storage1 = gt_storage.ones(
        default_parameters=backend,
        shape=semantic_shape,
        axes=axes1,
        dtype=dtype,
        default_origin=tuple([0] * ndims),
    )
    storage2 = gt_storage.ones(
        default_parameters=backend,
        shape=semantic_shape,
        axes=axes2,
        dtype=dtype,
        default_origin=tuple([0] * ndims),
    )
    array_masked_shape = np.ones(shape=storage1.shape, dtype=dtype)
    array_expanded_shape = np.ones(
        shape=tuple(semantic_shape[i] if mask2[i] else 1 for i in range(len(semantic_shape))),
        dtype=dtype,
    )

    array_all = np.ones(shape=semantic_shape, dtype=dtype)
    storage_all = gt_storage.ones(
        default_parameters=backend,
        shape=semantic_shape,
        dtype=dtype,
        default_origin=tuple([0] * len(semantic_shape)),
    )

    if any([m1 and m2 for m1, m2 in zip(mask1, mask2)]):
        storage_conflicting_shape = gt_storage.ones(
            default_parameters=backend,
            shape=tuple(s + 1 for s in storage2.shape),
            axes=axes2,
            dtype=dtype,
            default_origin=tuple([0] * len(semantic_shape)),
        )
        array_conflicting_shape = np.ones(shape=tuple(s + 1 for s in storage2.shape), dtype=dtype)
    # if not sum(mask1)==3:
    #     storage_conflicting_ndims = gt_storage.ones(
    #         default_parameters=backend,
    #         shape=storage1.shape,
    #         axes="IJK",
    #         dtype=dtype,
    #         default_origin=tuple([0] * len(semantic_shape)),
    #     )
    if sum(mask1) >= 2:
        array_underdetermined_shape = np.ones(shape=storage1.shape[: sum(mask1)], dtype=dtype)

    def _broadcast_shape():
        for i in range(ndims):
            assert (not mask1[i]) or storage1.shape[sum(mask1[: i + 1]) - 1] == semantic_shape[
                i
            ], "Bad shape in storage."
            assert (not mask2[i]) or storage2.shape[sum(mask2[: i + 1]) - 1] == semantic_shape[
                i
            ], "Bad shape in storage."
            if mask1[i]:
                yield storage1.shape[sum(mask1[: i + 1]) - 1]
            elif mask2[i]:
                yield storage2.shape[sum(mask2[: i + 1]) - 1]

    broadcast_shape = tuple(s for s in _broadcast_shape())

    res = storage1 + storage2
    assert all([res.mask[i] == (storage1.mask[i] or storage2.mask[i]) for i in range(ndims)])
    assert res.shape == broadcast_shape

    res = storage1 + array_expanded_shape
    assert all([res.mask[i] == storage1.mask[i] for i in range(ndims)])
    assert res.shape == broadcast_shape

    res = storage1 + array_masked_shape
    assert all([res.mask[i] == storage1.mask[i] for i in range(ndims)])
    assert res.shape == array_masked_shape.shape

    res = storage1 + storage_all
    assert all(res.mask)
    assert res.shape == semantic_shape
    res = storage_all + storage1
    assert all(res.mask)
    assert res.shape == semantic_shape

    res = storage1 + array_all
    assert all(res.mask)
    assert res.shape == semantic_shape
    res = array_all + storage1
    assert all(res.mask)
    assert res.shape == semantic_shape

    if any([m1 and m2 for m1, m2 in zip(mask1, mask2)]):
        with pytest.raises(ValueError):
            storage1 + storage_conflicting_shape
        with pytest.raises(ValueError):
            storage_conflicting_shape + storage1

        with pytest.raises(ValueError):
            storage1 + array_conflicting_shape
        with pytest.raises(ValueError):
            array_conflicting_shape + storage1
    #
    # with pytest.raises(ValueError):
    #     storage1 + storage_conflicting_ndims
    # with pytest.raises(ValueError):
    #     storage_conflicting_ndims + storage1
    if sum(mask1) >= 2:
        with pytest.raises(ValueError):
            storage1 + array_underdetermined_shape
        with pytest.raises(ValueError):
            array_underdetermined_shape + storage1


def generate_ufunc_input_left_shift(data, dtypes, shape):
    # TODO: generate data such that behavior is defined according to C standard
    #      (or at least consistent with numpy's implementation)
    return generate_ufunc_input(data, dtypes, shape)


def generate_ufunc_input_right_shift(data, dtypes, shape):
    # TODO: generate data such that behavior is defined according to C standard
    #      (or at least consistent with numpy's implementation)
    return generate_ufunc_input(data, dtypes, shape)


def generate_ufunc_input(data, dtypes, shape):
    arrays = []
    for dtype in dtypes:
        array = data.draw(hyp.extra.numpy.arrays(dtype=dtype, shape=shape), label="array")
        arrays.append(array)
    return arrays


UFUNC_DATA_GENERATOR = {
    "left_shift": generate_ufunc_input_left_shift,
    "ilshift": generate_ufunc_input_left_shift,
    "right_shift": generate_ufunc_input_right_shift,
    "irshift": generate_ufunc_input_right_shift,
}


@pytest.mark.parametrize(
    "function_name",
    # shift operations not tested since behavior is undefined for some inputs
    [
        f.__name__  # such that it is easy to see which tests fail
        for f in [
            ufunc
            for ufunc in Storage.SUPPORTED_UFUNCS
            if ufunc.__name__ not in ["left_shift", "right_shift"]
        ]
        + [
            op.iadd,
            op.iand,
            op.ifloordiv,
            # op.ilshift,
            op.imod,
            op.imul,
            op.ior,
            op.ipow,
            # op.irshift,
            op.isub,
            op.itruediv,
            op.ixor,
        ]
    ],
)
@hyp.given(data=hyp_st.data())
def test_ufunc_call(function_name, data):
    import gt4py.ir.nodes

    if function_name in np.core.umath.__dict__ and isinstance(
        np.core.umath.__dict__[function_name], np.ufunc
    ):
        function = np.core.umath.__dict__[function_name]
    elif function_name in op.__dict__:
        function = op.__dict__[function_name]
    else:
        raise KeyError

    shape = tuple(
        data.draw(
            hyp_st.lists(hyp_st.integers(min_value=1, max_value=3), min_size=3, max_size=3),
            label="shape",
        )
    )
    backend = (
        "gtmc"  # nothing is compiled, just so there is a non-trivial alignment, layout and padding
    )
    nin = function.nin if isinstance(function, np.ufunc) else 2
    dtype_st = hyp_st.one_of([hyp_st.just(v) for v in gt4py.ir.nodes.DataType.supported_dtypes])
    dtypes = []
    for i in range(nin):
        dtypes.append(data.draw(dtype_st))
    generate_data = UFUNC_DATA_GENERATOR.get(function_name, generate_ufunc_input)
    arrays = generate_data(data, dtypes, shape)
    inputs = []
    use_storage = data.draw(
        hyp_st.lists(hyp_st.booleans(), min_size=nin, max_size=nin).filter(
            lambda x: sum(x) > 0  # at least one storage in the input
        )
    )
    for i, array in enumerate(arrays):
        if use_storage[i]:
            inputs.append(
                gt_storage.storage(array, default_parameters=backend, default_origin=(1, 1, 1))
            )
        else:
            inputs.append(array)

    try:
        res = function(*inputs)
    except Exception as e:
        e_res = e
    else:
        e_res = None
    try:
        ref = function(*arrays)
    except Exception as e:
        e_ref = e
    else:
        e_ref = None

    if e_ref is not None or e_res is not None:
        assert str(e_res) == str(e_ref)
        assert type(e_res) == type(e_ref)
        return

    if not isinstance(res, tuple):
        assert not isinstance(ref, tuple)
        res = (res,)
        ref = (ref,)
    else:
        assert isinstance(ref, tuple)

    for ref_f, res_f in zip(ref, res):
        assert res_f.shape == ref_f.shape
        assert res_f.dtype == ref_f.dtype
        np.testing.assert_allclose(np.asarray(res_f), ref_f, equal_nan=True, atol=0.0, rtol=0.0)
        if isinstance(function, np.ufunc):
            assert isinstance(res_f, Storage)
        if isinstance(res_f, Storage):
            assert res_f.default_origin == (1,) * len(shape)
            assert (
                res_f._alignment
                == gt4py.storage.default_parameters.get_default_parameters(backend).alignment
            )


# @pytest.mark.parametrize(
#     "function_name",
#     [
#         f.__name__  # such that it is easy to see which tests fail
#         for f in list(Storage.SUPPORTED_UFUNCS)
#     ],
# )
# @hyp.given(hyp_data=hyp_st.data())
# def test_ufunc_reduce(function_name, hyp_data):
#     import gt4py.ir.nodes
#
#     ufunc = np.core.umath.__dict__[function_name]
#
#     # dtype_st = hyp_st.one_of([hyp_st.just(t) for t in gt4py.ir.nodes.DataType.supported_dtypes])
#     dtype_st = hyp_st.one_of([hyp_st.just(t) for t in gt4py.ir.nodes.DataType.supported_dtypes])
#     dtype_in = hyp_data.draw(dtype_st)
#     dtype_arg = hyp_data.draw(dtype_st)
#     dtype_out = hyp_data.draw(dtype_st)
#
#     data = hyp_data.draw(
#         hyp_np.arrays(
#             dtype=dtype_in,
#             shape=hyp_st.one_of(
#                 [hyp_st.just(v) for v in [(1, 1, 1), (3, 3, 3), (1, 1), (3, 3), (1,), (3,), ()]]
#             ),
#         )
#     )
#
#     # ufunc = hyp_data.draw(
#     #     # hyp_st.one_of([hyp_st.just(u) for u in Storage.SUPPORTED_UFUNCS])
#     #     hyp_st.one_of([hyp_st.just(np.add)])
#     # )
#
#     data_storage = gt_storage.from_array(data, backend="gtmc", default_origin=(0, 0, 0))
#
#     # a, axis, dtype, out, keepdims, initial, where
#
#     def _test(**kwargs):
#         axis = kwargs.get("axis", None)
#
#         out_ndarray = ufunc.reduce(data, **kwargs)
#         out_storage: Storage = ufunc.reduce(data_storage, **kwargs)
#         assert type(out_storage) == type(data_storage)
#         assert out_ndarray.shape == out_storage.shape
#         assert np.all(out_ndarray == out_storage)
#         if kwargs.get("keepdims", False):
#             assert out_storage.mask[axis]
#             assert not any(out_storage.mask[:axis])
#             assert not any(out_storage.mask[axis + 1 :])
#         else:
#             assert not out_storage.mask[axis]
#             assert all(out_storage.mask[:axis])
#             assert all(out_storage.mask[axis + 1 :])
#
#     _test(keepdims=True)
#     _test(keepdims=False)
#     _test()
#     for axis in range(len(data.shape)):
#         _test(keepdims=True, axis=axis)
#         _test(keepdims=False, axis=axis)
#         _test(axis=axis)


@pytest.mark.parametrize(
    "function_name",
    [
        f.__name__  # such that it is easy to see which tests fail
        for f in list(Storage.SUPPORTED_UFUNCS)
    ],
)
@hyp.given(data=hyp_st.data())
def test_ufunc_reduce(function_name, data):
    import gt4py.ir.nodes

    function = np.core.umath.__dict__[function_name]

    shape = tuple(
        data.draw(
            hyp_st.lists(hyp_st.integers(min_value=1, max_value=3), min_size=3, max_size=3),
            label="shape",
        )
    )
    backend = (
        "gtmc"  # nothing is compiled, just so there is a non-trivial alignment, layout and padding
    )

    axis = tuple(
        data.draw(
            hyp_st.lists(
                hyp_st.integers(min_value=0, max_value=len(shape)), min_size=0, max_size=len(shape)
            )
        )
    )

    nin = function.nin if isinstance(function, np.ufunc) else 2
    dtype = data.draw(
        hyp_st.one_of([hyp_st.just(v) for v in gt4py.ir.nodes.DataType.supported_dtypes])
    )
    generate_data = UFUNC_DATA_GENERATOR.get(function_name, generate_ufunc_input)
    array = generate_data(data, [dtype], shape)[0]

    input = gt_storage.storage(array, default_parameters=backend, default_origin=(1, 1, 1))

    try:
        res = function.reduce(input, axis=axis)
    except Exception as e:
        e_res = e
    else:
        e_res = None

    try:
        ref = function.reduce(array, axis=axis)
    except Exception as e:
        e_ref = e
    else:
        e_ref = None

    if e_ref is not None or e_res is not None:
        assert str(e_res) == str(e_ref)
        assert type(e_res) == type(e_ref)
        return

    if not isinstance(res, tuple):
        assert not isinstance(ref, tuple)
        res = (res,)
        ref = (ref,)
    else:
        assert isinstance(ref, tuple)

    for ref_f, res_f in zip(ref, res):
        assert res_f.shape == ref_f.shape
        assert res_f.dtype == ref_f.dtype
        np.testing.assert_allclose(res_f, ref_f, equal_nan=True, atol=0.0, rtol=0.0)
        if isinstance(function, np.ufunc):
            assert isinstance(res_f, Storage)
        if isinstance(res_f, Storage):
            assert res_f.default_origin == (1,) * len(res_f.shape)
            assert (
                res_f._alignment
                == gt4py.storage.default_parameters.get_default_parameters(backend).alignment
            )


#
#
# def test_ufunc_at():
#     pytest.skip()
#
#
# @pytest.mark.parametrize("function", [np.all, np.any])
# def test_allany(function, data):
#     data_storage = gt_storage.from_array(data, backend="gtmc", default_origin=(0, 0, 0))
#
#     def _test(**kwargs):
#         axis = kwargs.get("axis", None)
#
#         out_ndarray = function(data, **kwargs)
#         out_storage: Storage = function(data_storage, **kwargs)
#         assert type(out_storage) == type(data_storage)
#         assert out_ndarray.shape == out_storage.shape
#         assert np.all(out_ndarray == out_storage)
#
#     _test(data, keepdims=True)
#     _test(data, keepdims=False)
#     _test(data)
#     for axis in range(len(data.shape)):
#         _test(data, keepdims=True, axis=axis)
#         _test(data, keepdims=False, axis=axis)
#         _test(data, axis=axis)
#
#
# @hyp.given(data=hyp_st.data())
# def test_setitem(data):
#     pytest.skip("Test Not Implemented")
#
#
# @hyp.given(data=hyp_st.data())
# def test_getitem(data):
#     pytest.skip("Test Not Implemented")


@hyp.given(data=hyp_st.data())
def test_setitem(data):

    ndim_target = data.draw(hyp_st.integers(min_value=0, max_value=3))
    axes_target = sorted(
        data.draw(
            hyp_st.lists(
                hyp_st.sampled_from("IJK"), unique=True, min_size=ndim_target, max_size=ndim_target
            )
        )
    )
    ndim_value = data.draw(hyp_st.integers(min_value=0, max_value=ndim_target))
    axes_value = sorted(
        data.draw(
            hyp_st.lists(
                hyp_st.sampled_from(axes_target),
                unique=True,
                min_size=ndim_value,
                max_size=ndim_value,
            )
        )
    )
    axes_target = "".join(axes_target)
    axes_value = "".join(axes_value)

    target = gt_storage.zeros(
        dtype=np.float64,
        shape=(5,) * ndim_target,
        axes=axes_target,
        default_parameters="debug",
        default_origin=(0,) * ndim_target,
    )
    value = gt_storage.zeros(
        dtype=np.float64,
        shape=(5,) * ndim_value,
        axes=axes_value,
        default_parameters="debug",
        default_origin=(0,) * ndim_value,
    )
    target[...] = value
    assert np.all(
        np.asarray(target)[
            tuple(slice(None, None, None) if c in axes_value else 0 for c in axes_target)
        ]
        == np.asarray(value)
    )
