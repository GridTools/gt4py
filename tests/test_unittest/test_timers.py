import pprint
from typing import Any, Dict

import numpy as np

from gt4py import gtscript, storage


def advection_def(
    in_phi: gtscript.Field[float],
    in_u: gtscript.Field[float],
    in_v: gtscript.Field[float],
    out_phi: gtscript.Field[float],  # type: ignore  # noqa
):
    with computation(PARALLEL), interval(...):  # type: ignore  # noqa
        u = 0.5 * (in_u[-1, 0, 0] + in_u[0, 0, 0])
        flux_x = u[0, 0, 0] * (in_phi[-1, 0, 0] if u[0, 0, 0] > 0 else in_phi[0, 0, 0])
        v = 0.5 * (in_v[0, -1, 0] + in_v[0, 0, 0])
        flux_y = v[0, 0, 0] * (in_phi[0, -1, 0] if v[0, 0, 0] > 0 else in_phi[0, 0, 0])
        out_phi = (  # noqa
            in_phi - (flux_x[1, 0, 0] - flux_x[0, 0, 0]) - (flux_y[0, 1, 0] - flux_y[0, 0, 0])
        )


def diffusion_def(
    in_phi: gtscript.Field[float], out_phi: gtscript.Field[float], *, alpha: float  # type: ignore  # noqa
):
    with computation(PARALLEL), interval(...):  # type: ignore  # noqa
        lap1 = (
            -4 * in_phi[0, 0, 0]
            + in_phi[-1, 0, 0]
            + in_phi[1, 0, 0]
            + in_phi[0, -1, 0]
            + in_phi[0, 1, 0]
        )
        lap2 = -4 * lap1[0, 0, 0] + lap1[-1, 0, 0] + lap1[1, 0, 0] + lap1[0, -1, 0] + lap1[0, 1, 0]
        flux_x = lap2[1, 0, 0] - lap2[0, 0, 0]
        flux_y = lap2[0, 1, 0] - lap2[0, 0, 0]
        out_phi = in_phi + alpha * (  # noqa
            flux_x[0, 0, 0] - flux_x[-1, 0, 0] + flux_y[0, 0, 0] - flux_y[0, -1, 0]
        )


if __name__ == "__main__":
    backend = "gtx86"
    nx = ny = nz = 128

    in_phi = storage.from_array(
        np.random.rand(nx, ny, nz), backend=backend, default_origin=(0, 0, 0), dtype=float
    )
    in_u = storage.from_array(
        np.random.rand(nx, ny, nz), backend=backend, default_origin=(0, 0, 0), dtype=float
    )
    in_v = storage.from_array(
        np.random.rand(nx, ny, nz), backend=backend, default_origin=(0, 0, 0), dtype=float
    )
    tmp_phi = storage.empty(
        shape=(nx, ny, nz), backend=backend, default_origin=(1, 1, 0), dtype=float
    )
    out_phi = storage.empty(
        shape=(nx, ny, nz), backend=backend, default_origin=(3, 3, 0), dtype=float
    )
    alpha = 1 / 32

    advection = gtscript.stencil(backend=backend, definition=advection_def, rebuild=True)
    diffusion = gtscript.stencil(backend=backend, definition=diffusion_def, rebuild=True)

    exec_info: Dict[str, Dict[str, Any]] = {}
    for _ in range(10):
        advection(
            in_phi,
            in_u,
            in_v,
            tmp_phi,
            origin=(1, 1, 0),
            domain=(nx - 2, ny - 2, nz),
            exec_info=exec_info,
        )
        diffusion(
            in_phi,
            out_phi,
            alpha=alpha,
            origin=(3, 3, 0),
            domain=(nx - 6, ny - 6, nz),
            exec_info=exec_info,
        )
    pprint.pprint(exec_info)
