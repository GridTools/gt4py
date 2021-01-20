# -*- coding: utf-8 -*-
import numpy as np
import pytest

from gt4py import gtscript as gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval

from ..definitions import CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS


INTERNAL_CPU_BACKENDS = list(set(CPU_BACKENDS) & set(INTERNAL_BACKENDS))
INTERNAL_GPU_BACKENDS = list(set(GPU_BACKENDS) & set(INTERNAL_BACKENDS))


class TestConstructionEndToEnd:
    @pytest.mark.parametrize(
        "alloc_fun",
        [
            gt_storage.empty,
            gt_storage.ones,
            gt_storage.zeros,
            lambda shape, *args, **kwargs: gt_storage.full(shape, 3.0, *args, **kwargs),
        ],
    )
    class TestConstructionFromParameters:
        def run_test(self, alloc_fun, device, managed):
            stor = alloc_fun(
                device=device, managed=managed, dtype=np.float32, halo=(1, 2, 3), shape=(2, 4, 6)
            )
            assert stor.halo == ((1, 1), (2, 2), (3, 3))
            assert stor.shape == (2, 4, 6)
            assert stor.dtype == np.float32
            assert isinstance(stor, gt_storage.Storage)
            return stor

        def test_cpu(self, alloc_fun):
            res = self.run_test(alloc_fun, None, False)
            assert isinstance(res, gt_storage.definitions.CPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_only(self, alloc_fun):
            res = self.run_test(alloc_fun, "gpu", False)
            assert isinstance(res, gt_storage.definitions.GPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_cuda_managed(self, alloc_fun):
            res = self.run_test(alloc_fun, "gpu", "cuda")
            assert isinstance(res, gt_storage.definitions.CudaManagedGPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_gt4py_managed(self, alloc_fun):
            res = self.run_test(alloc_fun, "gpu", "gt4py")
            assert isinstance(res, gt_storage.definitions.ExplicitlyManagedGPUStorage)

    @pytest.mark.parametrize(
        ["alloc_fun", "kwargs"],
        [
            (gt_storage.empty_like, {}),
            (gt_storage.ones_like, {}),
            (gt_storage.zeros_like, {}),
            (lambda data, *args, **kwargs: gt_storage.full_like(data, 3.0, *args, **kwargs), {}),
            (gt_storage.storage, {}),
        ],
    )
    class TestConstructionFromTemplate:
        def run_test(self, alloc_fun, data):
            stor = alloc_fun(data=data)
            assert stor.shape == data.shape
            assert stor.dtype == data.dtype
            assert stor.halo == data.halo

            assert isinstance(stor, gt_storage.Storage)
            return stor

        def test_cpu(self, alloc_fun, kwargs):
            if "shape" not in kwargs:
                kwargs["shape"] = (2, 4, 6)
            data = gt_storage.empty(device=None, managed=False, **kwargs)
            res = self.run_test(alloc_fun, data)
            assert isinstance(res, gt_storage.definitions.CPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_only(self, alloc_fun, kwargs):
            if "shape" not in kwargs:
                kwargs["shape"] = (2, 4, 6)
            data = gt_storage.empty(device="gpu", managed=False, **kwargs)
            res = self.run_test(alloc_fun, data)
            assert isinstance(res, gt_storage.definitions.GPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_cuda_managed(self, alloc_fun, kwargs):
            if "shape" not in kwargs:
                kwargs["shape"] = (2, 4, 6)
            data = gt_storage.empty(device="gpu", managed="cuda", **kwargs)
            res = self.run_test(alloc_fun, data)
            assert isinstance(res, gt_storage.definitions.CudaManagedGPUStorage)

        @pytest.mark.requires_gpu
        def test_gpu_gt4py_managed(self, alloc_fun, kwargs):
            if "shape" not in kwargs:
                kwargs["shape"] = (2, 4, 6)
            data = gt_storage.empty(device="gpu", managed="gt4py", **kwargs)
            res = self.run_test(alloc_fun, data)
            assert isinstance(res, gt_storage.definitions.ExplicitlyManagedGPUStorage)


@pytest.mark.parametrize(
    "backend",
    INTERNAL_CPU_BACKENDS
    + [pytest.param(b, marks=[pytest.mark.requires_gpu]) for b in INTERNAL_GPU_BACKENDS],
)
@pytest.mark.parametrize(
    "alloc_funs",
    [
        pytest.param((gt_storage.empty, gt_storage.empty_like), id="empty"),
        pytest.param((gt_storage.ones, gt_storage.ones_like), id="ones"),
        pytest.param((gt_storage.zeros, gt_storage.zeros_like), id="zeros"),
        pytest.param(
            (
                lambda shape, *args, **kwargs: gt_storage.full(shape, 3.0, *args, **kwargs),
                lambda data, *args, **kwargs: gt_storage.full_like(data, 3.0, *args, **kwargs),
            ),
            id="full",
        ),
    ],
)
@pytest.mark.parametrize(
    ["device", "managed"],
    [
        ("cpu", False),
        pytest.param("gpu", False, marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "cuda", marks=[pytest.mark.requires_gpu]),
        pytest.param("gpu", "gt4py", marks=[pytest.mark.requires_gpu]),
    ],
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
def test_construction_from_template_roundtrip_compatible(
    backend, device, managed, alloc_funs, shape
):
    alloc, alloc_like = alloc_funs

    template = alloc(shape, device=device, managed=managed, defaults=backend)
    storage = alloc_like(template)
    assert type(template) is type(storage)

    @gtscript.stencil(backend=backend)
    def stencil(field: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            field = 3.0  # noqa: F841

    stencil(storage)
