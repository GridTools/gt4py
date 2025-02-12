# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest

from gt4py._core.definitions import DeviceType


try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py
from gt4py.cartesian import gtscript
from gt4py.storage.cartesian.utils import allocate_cpu, allocate_gpu, normalize_storage_spec


CPU_LAYOUTS = [
    name for name, info in gt4py.storage.layout.REGISTRY.items() if info["device"] == DeviceType.CPU
]
GPU_LAYOUTS = [
    pytest.param(name, marks=pytest.mark.requires_gpu)
    for name, info in gt4py.storage.layout.REGISTRY.items()
    if info["device"] == DeviceType.CUDA
]

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
    aligned_index_strats = []

    for _i in range(dimension):
        shape_strats = shape_strats + [hyp_st.integers(min_value=1, max_value=64)]
    shape = draw(hyp_st.tuples(*shape_strats))
    for i in range(dimension):
        aligned_index_strats = aligned_index_strats + [
            hyp_st.integers(min_value=0, max_value=min(32, shape[i] - 1))
        ]

    aligned_index = draw(hyp_st.tuples(*aligned_index_strats))
    layout_order = draw(hyp_st.permutations(tuple(range(dimension))))
    return dict(
        dtype=dtype,
        alignment=alignment,
        shape=shape,
        aligned_index=aligned_index,
        layout_order=layout_order,
    )


@hyp_st.composite
def dimensions_strategy(draw):
    dimension = draw(hyp_st.integers(min_value=1, max_value=6))

    shape_strats = [hyp_st.integers(min_value=1, max_value=32)] * dimension
    shape = draw(hyp_st.tuples(*shape_strats))
    aligned_index_strats = [
        hyp_st.integers(min_value=0, max_value=min(32, shape[i] - 1)) for i in range(dimension)
    ]
    aligned_index = draw(hyp_st.tuples(*aligned_index_strats))

    mask_values = [True] * dimension
    if dimension < 3:
        mask_values += [False] * (3 - dimension)

    mask = draw(hyp_st.one_of(hyp_st.just(None), hyp_st.permutations(mask_values)))
    if mask is not None:
        select_dimensions = ["I", "J", "K"] + [str(d) for d in range(max(0, dimension - 3))]
        assert len(select_dimensions) == len(mask)
        dimensions = [d for m, d in zip(mask, select_dimensions) if m]
    else:
        dimensions = None
    return dict(shape=shape, aligned_index=aligned_index, dimensions=dimensions)


# ---- Tests ----
@hyp.given(param_dict=allocation_strategy())
def test_allocate_cpu(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    aligned_index = param_dict["aligned_index"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]

    raw_buffer, field = allocate_cpu(shape, layout_map, dtype, alignment_bytes, aligned_index)

    # check that memory of field is contained in raw_buffer
    np_byte_bounds = (
        np.byte_bounds if hasattr(np, "byte_bounds") else np.lib.array_utils.byte_bounds
    )
    assert (
        np_byte_bounds(field)[0] >= np_byte_bounds(raw_buffer)[0]
        and np_byte_bounds(field)[1] <= np_byte_bounds(raw_buffer)[1]
    )

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    for _i in range(100):
        slices = []
        for hidx in range(len(shape)):
            if hidx == np.argmax(layout_map):
                slices = slices + [slice(aligned_index[hidx], None)]
            else:
                slices = slices + [slice(random.randint(0, shape[hidx]), None)]
        assert field[tuple(slices)].ctypes.data % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    slices = []
    for _hidx in range(len(shape)):
        slices = slices + [0]
    field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [aligned_index[hidx]]
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
        raise AssertionError()

    # check if shape is properly set
    assert field.shape == shape


# TODO(egparedes): keep this and the previous tests separated just to differentiate
# between gpu and cpu tests, but factorize all common code in a single utility
# function which gets called from the test functions bodies.
@pytest.mark.requires_gpu
@hyp.given(param_dict=allocation_strategy())
def test_allocate_gpu(param_dict):
    alignment_bytes = param_dict["alignment"]
    dtype = param_dict["dtype"]
    aligned_index = param_dict["aligned_index"]
    shape = param_dict["shape"]
    layout_map = param_dict["layout_order"]
    device_raw_buffer, device_field = allocate_gpu(
        shape, layout_map, dtype, alignment_bytes, aligned_index
    )

    # Would like to check device_field.base against device_raw_buffer but
    # as_strided returns an ndarray where device_field.base is set to None.
    # Instead, check that the memory of field is contained in raws buffer
    assert (
        device_field.data.ptr >= device_raw_buffer.data.ptr
        and device_field[-1:].data.ptr <= device_raw_buffer[-1:].data.ptr
    )

    # check if the first compute-domain point in the last dimension is aligned for 100 random "columns"
    import random

    for _i in range(100):
        slices = []
        for hidx in range(len(shape)):
            if hidx == np.argmax(layout_map):
                slices = slices + [slice(aligned_index[hidx], None)]
            else:
                slices = slices + [slice(random.randint(0, shape[hidx]), None)]
        assert device_field[tuple(slices)].data.ptr % alignment_bytes == 0

    # check that writing does not give errors, e.g. because of going out of bounds
    slices = []
    for _hidx in range(len(shape)):
        slices = slices + [0]
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [aligned_index[hidx]]
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx] - 1]
    device_field[tuple(slices)] = 1

    slices = []
    for hidx in range(len(shape)):
        slices = slices + [shape[hidx]]

    with pytest.raises(IndexError):
        device_field[tuple(slices)] = 1

    # check if shape is properly set
    assert device_field.shape == shape


class TestNormalizeStorageSpec:
    def test_normal(self):
        aligned_index = (0, 0, 0)
        shape = (10, 10, 10)
        dtype = np.float64
        dimensions = ("I", "J", "K")

        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            aligned_index, shape, dtype, dimensions
        )
        assert aligned_index_out == aligned_index
        assert shape_out == shape
        assert dtype_out == dtype
        assert dimensions_out == dimensions

    def test_aligned_index(self):
        shape = (10, 10, 10)
        dtype = np.float64
        dimensions = ("I", "J", "K")

        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            (1, 1, 1), shape, dtype, dimensions
        )
        assert aligned_index_out == (1, 1, 1)
        assert shape_out == shape
        assert dtype_out == dtype
        assert dimensions_out == dimensions

        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            None, shape, dtype, dimensions
        )
        assert aligned_index_out == (0, 0, 0)

        with pytest.raises(TypeError, match="aligned_index"):
            normalize_storage_spec(("1", "1", "1"), shape, dtype, dimensions)
        with pytest.raises(ValueError, match="aligned_index"):
            normalize_storage_spec((1, 1, 1, 1), shape, dtype, dimensions)
        with pytest.raises(ValueError, match="aligned_index"):
            normalize_storage_spec((-1, -1, -1), shape, dtype, dimensions)

    def test_shape(self):
        aligned_index = (0, 0)
        dtype = np.float64
        dimensions = ("I", "J", "K")

        # Shape
        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            aligned_index, (10, 10), dtype, ("I", "J")
        )
        assert aligned_index_out == (0, 0)
        assert shape_out == (10, 10)
        assert dtype_out == dtype
        assert dimensions_out == ("I", "J")

        with pytest.raises(ValueError, match="non-matching"):
            normalize_storage_spec(aligned_index, (10, 20), dtype, ("J",))
        with pytest.raises(TypeError, match="shape"):
            normalize_storage_spec(aligned_index, "(10,10)", dtype, dimensions)
        with pytest.raises(TypeError, match="shape"):
            normalize_storage_spec(aligned_index, None, dtype, dimensions)
        with pytest.raises(ValueError, match="shape"):
            normalize_storage_spec(aligned_index, (10, 10, 0), dtype, dimensions)

    def test_dimensions(self):
        aligned_index = (0, 0)
        dtype = np.float64

        # Mask
        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            aligned_index, (10, 10), dtype, ("I", "J")
        )
        assert aligned_index_out == (0, 0)
        assert shape_out == (10, 10)
        assert dtype_out == dtype
        assert dimensions_out == ("I", "J")

        _, __, ___, dimensions_out = normalize_storage_spec(aligned_index, (10, 10), dtype, "IJ")
        assert dimensions_out == ("I", "J")

        _, __, ___, dimensions_out = normalize_storage_spec(
            aligned_index, (10, 10), dtype, gtscript.IJ
        )
        assert dimensions_out == ("I", "J")

        _, __, ___, dimensions_out = normalize_storage_spec(
            (0, 0, 0), (10, 10, 10), dtype, gtscript.IJK
        )
        assert dimensions_out == ("I", "J", "K")

        with pytest.raises(ValueError, match="dimensions"):
            normalize_storage_spec(aligned_index, (10, 10), dtype, ())

    def test_dtype(self):
        aligned_index = (0, 0, 0)
        shape = (10, 10, 10)
        dtype = np.float64
        dimensions = ("I", "J", "K")

        aligned_index_out, shape_out, dtype_out, dimensions_out = normalize_storage_spec(
            aligned_index, shape, (dtype, (2,)), dimensions
        )

        assert aligned_index_out == tuple([*aligned_index, 0])
        assert shape_out == tuple([*shape, 2])
        assert dtype_out == dtype
        assert dimensions_out == tuple([*dimensions, "0"])


def _create_ndarray_for_test_fill(*, dtype, aligned_index, shape, backend):
    return gt4py.storage.full(shape, 7, dtype, backend=backend, aligned_index=aligned_index)


def _create_ndarray_for_test_from_array(*, dtype, aligned_index, shape, backend):
    return gt4py.storage.from_array(
        np.ones(shape, dtype=dtype) * 7, dtype, backend=backend, aligned_index=aligned_index
    )


@pytest.fixture(
    params=[
        gt4py.storage.empty,
        gt4py.storage.ones,
        gt4py.storage.zeros,
        _create_ndarray_for_test_fill,
        _create_ndarray_for_test_from_array,
    ]
)
def alloc_fun(request):
    return request.param


@pytest.mark.parametrize("backend", CPU_LAYOUTS)
def test_cpu_constructor(alloc_fun, backend):
    stor = alloc_fun(dtype=np.float64, aligned_index=(1, 2, 3), shape=(2, 4, 6), backend=backend)
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, np.ndarray)


@pytest.mark.parametrize("backend", CPU_LAYOUTS)
def test_cpu_constructor_0d(alloc_fun, backend):
    stor = alloc_fun(shape=(), dtype=np.float64, backend=backend, aligned_index=())
    assert stor.shape == ()
    assert isinstance(stor, np.ndarray)


@pytest.mark.parametrize("backend", GPU_LAYOUTS)
def test_gpu_constructor(alloc_fun, backend):
    stor = alloc_fun(dtype=np.float64, aligned_index=(1, 2, 3), shape=(2, 4, 6), backend=backend)
    assert stor.shape == (2, 4, 6)
    assert isinstance(stor, cp.ndarray)


@pytest.mark.parametrize("backend", GPU_LAYOUTS)
def test_gpu_constructor_0d(alloc_fun, backend):
    stor = alloc_fun(shape=(), dtype=np.float64, backend=backend, aligned_index=())
    assert stor.shape == ()
    assert isinstance(stor, cp.ndarray)


@pytest.mark.requires_gpu
@hyp.given(param_dict=dimensions_strategy())
def test_masked_storage_gpu(param_dict):
    dimensions = param_dict["dimensions"]
    aligned_index = param_dict["aligned_index"]
    shape = param_dict["shape"]

    # no assert when all is defined in descriptor, no grid_group
    array = gt4py.storage.empty(
        dtype=np.float64,
        aligned_index=aligned_index,
        shape=shape,
        dimensions=dimensions,
        backend="gt:gpu",
    )

    assert isinstance(array, cp.ndarray)
    if dimensions is None:
        dimensions = ["I", "J", "K"][: len(shape)]
        if len(shape) > 3:
            dimensions += [str(d) for d in range(len(shape) - 3)]
    assert len(dimensions) == array.ndim
    assert len(dimensions) == len(array.shape)


def test_masked_storage_asserts():
    aligned_index = (1, 1, 1)
    shape = (2, 2, 2)
    backend = "gt:cpu_kfirst"

    with pytest.raises(ValueError):
        gt4py.storage.empty(
            dtype=np.float64,
            aligned_index=aligned_index,
            shape=shape,
            dimensions=(),
            backend=backend,
        )


def test_non_existing_backend():
    with pytest.raises(RuntimeError, match="backend"):
        gt4py.storage.empty(
            shape=[10, 10, 10],
            backend="non_existing_backend",
            aligned_index=[0, 0, 0],
            dtype=(np.float64, (3,)),
        )
