import numpy as np

from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval


# backend = "gtx86"  # "debug", "numpy", "gtx86", "gtcuda"
backend = "gtc:gt"  # "debug", "numpy", "gtx86", "gtcuda"

dtype = np.float64


# this decorator triggers compilation of the stencil
@gtscript.lazy_stencil(backend=backend, rebuild=True)
def demo_copy(in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(0, 1):
        out_field = 2.0  # noqa: F841 [assigned but not used]
    with computation(PARALLEL), interval(1, -1):
        out_field = in_field  # noqa: F841 [assigned but not used]
    with computation(PARALLEL), interval(-1, None):
        out_field = 3.0  # noqa: F841 [assigned but not used]


if __name__ == "__main__":
    N = 30
    shape = [N] * 3
    origin = (0, 0, 0)

    in_data = np.ones(shape)
    out_data = np.zeros(shape)

    in_storage = gt_storage.from_array(in_data, backend, default_origin=origin, dtype=dtype)
    out_storage = gt_storage.from_array(
        out_data,
        backend,
        default_origin=origin,
        dtype=dtype,
    )

    demo_copy(in_storage, out_storage)

    assert out_storage[1, 1, 0] == 2.0
    assert out_storage[1, 1, 1] == 1.0
    assert out_storage[1, 1, 29] == 3.0
