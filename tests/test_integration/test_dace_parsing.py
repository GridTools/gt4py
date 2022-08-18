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
from gt4py.storage import utils as storage_utils

from ..storage_test_utils import NdarraySubclassOriginWrapper, OriginWrapper


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

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    inp = 7.0

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_stencil_object(locoutp, locinp):
        defn(locoutp, par=locinp)

    call_stencil_object(locoutp=outp, locinp=inp)
    outp = storage_utils.cpu_copy(outp)

    assert np.allclose(outp, 7.0)


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting_frozen(dace_stencil, domain, outp_origin):
    backend = dace_stencil.backend
    frozen_stencil = dace_stencil.freeze(
        domain=domain, origin={"inp": (0, 0, 0), "outp": outp_origin}
    )

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    inp = wrapper_type(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )

    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)

    assert np.allclose(inp, 7.0)

    assert np.allclose(
        outp[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )

    assert np.sum(outp, axis=(0, 1, 2)) == np.prod(domain) * 7.0


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting_nofrozen(dace_stencil, domain, outp_origin):

    if domain == (3, 3, 3) and outp_origin == (7, 7, 7) and dace_stencil.backend == "daint:gpu":
        pytest.skip("Random failures on daint-ci, see github issue #848.")

    backend = dace_stencil.backend

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    inp = wrapper_type(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    origin = {"inp": (0, 0, 0), "outp": outp_origin}

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_stencil_object():
        dace_stencil(inp=inp, outp=outp, domain=domain, origin=origin)

    call_stencil_object()

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)

    assert np.allclose(inp, 7.0)
    assert np.allclose(
        outp[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )
    assert np.sum(outp, axis=(0, 1, 2)) == np.prod(domain) * 7.0


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

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    inp = wrapper_type(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)

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

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    inp = wrapper_type(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )
    unused_field = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil():
        frozen_stencil(inp=inp, unused_field=unused_field, outp=outp, unused_par=7.0)

    call_frozen_stencil()

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_optional_arg_provide_aot(backend):
    import dace.data

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

    # DaCe in the currently tagged version of this repo (~=0.13 doesn't support non-numpy arrays)
    wrapper_type = OriginWrapper if "gpu" in backend else NdarraySubclassOriginWrapper

    inp = wrapper_type(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )
    unused_field = wrapper_type(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_frozen_stencil(
        inp: dace.data.create_datadescriptor(inp),
        outp: dace.data.create_datadescriptor(outp),
        unused_field: dace.data.create_datadescriptor(unused_field),
        unused_par: dace.float64,  # type: ignore
    ):
        frozen_stencil(inp=inp, unused_field=unused_field, outp=outp, unused_par=unused_par)

    csdfg = call_frozen_stencil.compile()
    csdfg(inp=inp, outp=outp, unused_field=unused_field, unused_par=7.0)

    inp = storage_utils.cpu_copy(inp)
    outp = storage_utils.cpu_copy(outp)

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

    inp = NdarraySubclassOriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            dtype=np.float64,
            shape=(10, 10, 10),
            aligned_index=(0, 0, 0),
            backend="numpy",
        ),
        origin=(0, 0, 0),
    )
    outp = NdarraySubclassOriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64,
            shape=(10, 10, 10),
            aligned_index=(0, 0, 0),
            backend="numpy",
        ),
        origin=(0, 0, 0),
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
