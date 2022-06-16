import numpy as np
import pytest

from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.dace_lazy_stencil import DaCeLazyStencil
from gt4py.gtscript import PARALLEL, computation, interval
from gt4py.stencil_builder import StencilBuilder


dace = pytest.importorskip("dace")


def simple_stencil_defn(outp: gtscript.Field[np.float64], par: np.float64):
    with computation(PARALLEL), interval(...):
        outp = par  # noqa F841: local variable 'outp' is assigned to but never used


@pytest.mark.parametrize(
    "backend",
    ["dace:cpu", pytest.param("dace:gpu", marks=[pytest.mark.requires_gpu])],
)
def test_lazy_sdfg(backend):

    builder = StencilBuilder(simple_stencil_defn, backend="dace:cpu").with_options(
        name="simple_stencil", module=simple_stencil_defn.__module__
    )
    lazy_s = DaCeLazyStencil(builder)

    outp = gt_storage.zeros(
        dtype=np.float64, shape=(10, 10, 10), default_origin=(0, 0, 0), backend=backend
    )

    inp = 7.0

    outp.host_to_device()

    @dace.program(device=dace.DeviceType.GPU if "gpu" in backend else dace.DeviceType.CPU)
    def call_lazy_s(locoutp, locinp):
        lazy_s(locoutp, par=locinp)

    call_lazy_s.compile(locoutp=outp, locinp=inp)

    assert "implementation" not in lazy_s.__dict__
