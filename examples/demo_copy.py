import numpy as np

import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage


backend = "gtc:py"  # "debug", "numpy", "gtx86", "gtcuda"
# backend = "debug"  # "debug", "numpy", "gtx86", "gtcuda"
dtype = np.float64


# this decorator triggers compilation of the stencil
@gtscript.lazy_stencil(backend=backend, rebuild=True)
def demo_copy(in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_field = in_field
        out_field = in_field


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

print(out_storage[1, 1, 1])
