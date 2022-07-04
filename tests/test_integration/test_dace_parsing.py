# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import re

import hypothesis.strategies as hyp_st
import numpy as np
import pytest

import gt4py.config
from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval


dace = pytest.importorskip("dace")


pytestmark = pytest.mark.usefixtures("dace_env")


@pytest.fixture(scope="module")
def dace_env():
    gt_cache_path = (
        pathlib.Path(gt4py.config.cache_settings["root_path"])
        / gt4py.config.cache_settings["dir_name"]
        / "dacecache"
    )
    with dace.config.temporary_config():
        # Setting max_concurrent_streams to -1 configures dace to only use the default stream.
        dace.config.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)
        dace.config.Config.set("compiler", "cpu", "openmp_sections", value=False)
        dace.config.Config.set("compiler", "cpu", "args", value="")
        dace.config.Config.set("compiler", "allow_view_arguments", value=True)
        dace.config.Config.set("default_build_folder", value=str(gt_cache_path))
        yield


@pytest.fixture(
    params=[
        "dace:cpu",
        pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu]),
    ]
)
def dace_stencil(request):
    @gtscript.stencil(backend=request.param)
    def defn(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    return defn


def tuple_st(min_value, max_value):
    return hyp_st.tuples(
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
    )


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_basic(backend):
    @gtscript.stencil(backend=backend)
    def defn(outp: gtscript.Field[np.float64], par: np.float64):
        with computation(PARALLEL), interval(...):
            outp = par  # noqa F841: local variable 'outp' is assigned to but never used

    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )

    inp = 7.0

    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_stencil_object(locoutp, locinp):
        defn(locoutp, par=locinp)

    call_stencil_object(locoutp=outp, locinp=inp)
    outp.device_to_host(force=True)
    assert np.allclose(outp, 7.0)


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting_frozen(dace_stencil, domain, outp_origin):
    backend = dace_stencil.backend
    frozen_stencil = dace_stencil.freeze(
        domain=domain, origin={"inp": (0, 0, 0), "outp": outp_origin}
    )
    inp = gt_storage.from_array(
        data=7.0, dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )

    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )

    inp.host_to_device()
    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    outp.device_to_host(force=True)

    assert np.allclose(inp, 7.0)

    assert np.allclose(
        np.asarray(outp)[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )

    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == np.prod(domain) * 7.0


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting_nofrozen(dace_stencil, domain, outp_origin):

    if domain == (3, 3, 3) and outp_origin == (7, 7, 7) and dace_stencil.backend == "daint:gpu":
        pytest.skip("Random failures on daint-ci, see github issue #848.")

    backend = dace_stencil.backend
    inp = gt_storage.from_array(
        data=7.0, dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )

    origin = {"inp": (0, 0, 0), "outp": outp_origin}

    inp.host_to_device()
    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_stencil_object():
        dace_stencil(inp=inp, outp=outp, domain=domain, origin=origin)

    call_stencil_object()

    outp.device_to_host(force=True)

    assert np.allclose(inp, 7.0)
    assert np.allclose(
        np.asarray(outp)[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == np.prod(domain) * 7.0


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_optional_arg_noprovide(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10),
        origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)},
    )

    inp = gt_storage.from_array(
        data=7.0, dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    inp.host_to_device()
    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    outp.device_to_host(force=True)

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_optional_arg_provide(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10),
        origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)},
    )

    inp = gt_storage.from_array(
        data=7.0, dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    unused_field = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    inp.host_to_device()
    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, unused_field=unused_field, outp=outp, unused_par=7.0)

    call_frozen_stencil()

    outp.device_to_host(force=True)

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_optional_arg_provide_aot(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10),
        origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)},
    )

    inp = gt_storage.from_array(
        data=7.0, dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    unused_field = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )
    inp.host_to_device()
    outp.host_to_device()

    storage = dace.StorageType.GPU_Global if "gpu" in backend else dace.StorageType.CPU_Heap

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil(
        inp: dace.data.Array(
            shape=inp.shape,
            strides=tuple(s // inp.itemsize for s in inp.strides),
            dtype=dace.float64,
            storage=storage,
        ),
        outp: dace.data.Array(
            shape=outp.shape,
            strides=tuple(s // outp.itemsize for s in outp.strides),
            dtype=dace.float64,
            storage=storage,
        ),
        unused_field: dace.data.Array(
            shape=unused_field.shape,
            strides=tuple(s // unused_field.itemsize for s in unused_field.strides),
            dtype=dace.float64,
            storage=storage,
        ),  # type: ignore
        unused_par: dace.float64,  # type: ignore
    ):
        frozen_stencil(inp=inp, unused_field=unused_field, outp=outp, unused_par=unused_par)

    csdfg = call_frozen_stencil.compile()
    csdfg(inp=inp, outp=outp, unused_field=unused_field, unused_par=7.0)

    outp.device_to_host(force=True)

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


def test_nondace_raises():
    @gtscript.stencil(backend="numpy")
    def numpy_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = numpy_stencil.freeze(
        domain=(3, 3, 3), origin={"inp": (0, 0, 0), "outp": (0, 0, 0)}
    )

    inp = gt_storage.from_array(
        data=7.0,
        dtype=np.float64,
        shape=(10, 10, 10),
        default_origin=(0, 0, 0),
        backend="numpy",
    )
    outp = gt_storage.zeros(
        dtype=np.float64,
        shape=(10, 10, 10),
        default_origin=(0, 0, 0),
        backend="numpy",
    )

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Only dace backends are supported in DaCe-orchestrated programs." ' (found "numpy")'
        ),
    ):
        call_frozen_stencil()
