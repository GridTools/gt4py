# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dace
else:
    dace = pytest.importorskip("dace")

import hypothesis.strategies as hyp_st
import numpy as np
import pathlib
import re
import typing

from gt4py import cartesian as gt4pyc, storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, computation, interval
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.storage.cartesian import utils as storage_utils
from gt4py.cartesian.backend.dace_lazy_stencil import DaCeLazyStencil

from cartesian_tests.utils import OriginWrapper

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable add the marker to all test functions in this module.
pytestmark = [pytest.mark.requires_dace, pytest.mark.usefixtures("dace_env")]


@pytest.fixture(scope="module")
def dace_env():
    gt_cache_path = (
        pathlib.Path(gt4pyc.config.cache_settings["root_path"])
        / gt4pyc.config.cache_settings["dir_name"]
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


@pytest.fixture(params=[gtscript.stencil, gtscript.lazy_stencil])
def decorator(request):
    return request.param


def tuple_st(min_value, max_value):
    return hyp_st.tuples(
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
    )


@pytest.mark.parametrize(
    "backend", ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])]
)
def test_basic(decorator, backend):
    @decorator(backend=backend)
    def defn(outp: gtscript.Field[np.float64], par: np.float64):
        with computation(PARALLEL), interval(...):
            outp = par  # noqa: F841 [unused-variable]

    outp = OriginWrapper(
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
def test_origin_offsetting_frozen(domain, outp_origin):
    backend = "dace:cpu"

    @gtscript.stencil(backend=backend)
    def dace_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    frozen_stencil = dace_stencil.freeze(
        domain=domain, origin={"inp": (0, 0, 0), "outp": outp_origin}
    )

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    assert np.allclose(inp, 7.0)

    assert np.allclose(
        np.asarray(outp)[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )

    assert np.sum(outp, axis=(0, 1, 2)) == np.prod(domain) * 7.0


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting_nofrozen(domain, outp_origin):
    backend = "dace:cpu"

    @gtscript.stencil(backend=backend)
    def dace_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    origin = {"inp": (0, 0, 0), "outp": outp_origin}

    @dace.program
    def call_stencil_object():
        dace_stencil(inp=inp, outp=outp, domain=domain, origin=origin)

    call_stencil_object()

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
def test_origin_offsetting_nofrozen_default_origin(domain, outp_origin):
    backend = "dace:cpu"

    @gtscript.stencil(backend=backend)
    def dace_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            dtype=np.float64,
            shape=(10, 10, 10),
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=outp_origin, backend=backend
        ),
        origin=outp_origin,
    )

    @dace.program
    def call_stencil_object(locinp, locoutp):
        dace_stencil(inp=locinp, outp=locoutp, domain=domain)

    call_stencil_object(locinp=inp, locoutp=outp)

    assert np.allclose(np.asarray(inp), 7.0)
    assert np.allclose(
        np.asarray(outp)[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == np.prod(domain) * 7.0


def test_optional_arg_noprovide():
    backend = "dace:cpu"

    @gtscript.stencil(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10), origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)}
    )

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


def test_optional_arg_provide(decorator):
    backend = "dace:cpu"

    @decorator(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )
    unused_field = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    # @dace.program
    def call_stencil():
        stencil(
            inp=inp,
            unused_field=unused_field,
            outp=outp,
            unused_par=7.0,
            domain=(3, 3, 10),
            origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)},
        )

    call_stencil()

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


def test_optional_arg_provide_aot(decorator):
    backend = "dace:cpu"

    @decorator(backend=backend)
    def stencil(
        inp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            shape=(10, 10, 10),
            dtype=np.float64,
            aligned_index=(0, 0, 0),
            backend=backend,
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )
    unused_field = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
        ),
        origin=(0, 0, 0),
    )

    @dace.program
    def call_stencil(
        inp: dace.data.create_datadescriptor(inp),
        outp: dace.data.create_datadescriptor(outp),
        unused_field: dace.data.create_datadescriptor(unused_field),
        unused_par: dace.float64,  # type: ignore
    ):
        stencil(
            inp=inp,
            unused_field=unused_field,
            outp=outp,
            unused_par=unused_par,
            domain=(3, 3, 10),
            origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)},
        )

    csdfg = call_stencil.compile()
    csdfg(inp=inp, outp=outp, unused_field=unused_field, unused_par=7.0)

    assert np.allclose(inp, 7.0)
    assert np.allclose(np.asarray(outp)[2:5, 2:5, :], 7.0)
    assert np.sum(np.asarray(outp), axis=(0, 1, 2)) == 90 * 7.0


def test_nondace_raises(decorator):
    @decorator(backend="numpy")
    def numpy_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa: F841 [unused-variable]

    inp = OriginWrapper(
        array=gt_storage.full(
            fill_value=7.0,
            dtype=np.float64,
            shape=(10, 10, 10),
            aligned_index=(0, 0, 0),
            backend="numpy",
        ),
        origin=(0, 0, 0),
    )
    outp = OriginWrapper(
        array=gt_storage.zeros(
            dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend="numpy"
        ),
        origin=(0, 0, 0),
    )

    @dace.program
    def call_stencil():
        numpy_stencil(
            inp=inp, outp=outp, domain=(3, 3, 3), origin={"inp": (0, 0, 0), "outp": (0, 0, 0)}
        )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Only dace backends are supported in DaCe-orchestrated programs." ' (found "numpy")'
        ),
    ):
        call_stencil()


@typing.no_type_check
def simple_stencil_defn(outp: gtscript.Field[np.float64], par: np.float64):
    with computation(PARALLEL), interval(...):
        outp = par  # noqa: F841 [unused-variable]


def test_lazy_sdfg():
    backend = "dace:cpu"
    builder = StencilBuilder(simple_stencil_defn, backend="dace:cpu").with_options(
        name="simple_stencil", module=simple_stencil_defn.__module__
    )
    lazy_s = DaCeLazyStencil(builder)

    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), aligned_index=(0, 0, 0), backend=backend
    )

    inp = 7.0

    @dace.program
    def call_lazy_s(locoutp, locinp):
        lazy_s(locoutp, par=locinp)

    call_lazy_s.compile(locoutp=outp, locinp=inp)

    assert "implementation" not in lazy_s.__dict__
