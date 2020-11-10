import numpy as np

from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval


backend = "gtc:py"  # "debug", "numpy", "gtx86", "gtcuda"
dtype = np.float64


# this decorator triggers compilation of the stencil
@gtscript.stencil(backend=backend, rebuild=True)
def demo_lap(in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_field = (  # noqa: F841 [assigned but not used]
            -4 * in_field
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, -0]
            + in_field[0, 1, 0]
        )


N = 30
shape = [N] * 3
origin = (1, 1, 0)

in_data = np.ones(shape)
out_data = np.zeros(shape)

in_storage = gt_storage.from_array(in_data, backend, default_origin=origin, dtype=dtype)
out_storage = gt_storage.from_array(
    out_data,
    backend,
    default_origin=origin,
    dtype=dtype,
)

demo_lap(in_storage, out_storage)

print(out_storage[1, 1, 1])
