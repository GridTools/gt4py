# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Shallow Water Model using the Python Array API standard.

This implementation uses symmetric halo lines (1 on each side) for all fields.
Periodic boundary conditions are applied via halo exchange.

Compatible with any array library supporting the Array API standard:
  numpy, jax.numpy, cupy, array_api_strict, etc.

Usage:
  python swm_array_api.py --array-library numpy
  python swm_array_api.py --array-library jax
  python swm_array_api.py --array-library torch
  python swm_array_api.py --array-library cupy
  python swm_array_api.py --strict             # validate compliance with array_api_strict wrapping
  python swm_array_api.py --array-library jax --compile    # run with jax.jit
  python swm_array_api.py --array-library torch --compile  # run with torch.compile
  python swm_array_api.py --array-library torch --compile --device cuda  # torch.compile on GPU
  python swm_array_api.py --array-library jax --compile --device cpu     # jax.jit on CPU
"""

import argparse
from time import perf_counter
from array_api_compat import array_namespace
import initial_conditions


def _get_array_module(name):
    """Import and return the array module for the given library name."""
    if name == "numpy":
        import numpy

        return numpy
    elif name == "jax":
        import jax.numpy

        return jax.numpy
    elif name == "torch":
        import torch

        return torch
    elif name == "cupy":
        import cupy

        return cupy
    elif name == "array_api_strict":
        import array_api_strict

        return array_api_strict
    else:
        raise ValueError(f"Unknown array library: {name}")


def _to_numpy(arr):
    """Convert array to numpy, handling GPU/CUDA tensors."""
    import numpy as np

    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(arr)


def _interior_to_halo(xp, interior):
    """Build (M+2, N+2) array from (M, N) interior with periodic halos.

    Wraps the interior periodically: last col -> left halo, first col -> right halo,
    last row -> top halo, first row -> bottom halo.
    """
    middle_rows = xp.concat([interior[:, -1:], interior, interior[:, :1]], axis=1)
    return xp.concat([middle_rows[-1:, :], middle_rows, middle_rows[:1, :]], axis=0)


# ---------------------------------------------------------------------------
# Stencil building blocks following Sadourny (1975), J. Atm. Sci. 32:680–688.
#
# The model uses an Arakawa C-grid with four types of grid point:
#   p-point  at (i,   j  )  —  pressure/height h
#   u-point  at (i+½, j  )  —  zonal velocity u      (staggered ½-cell in x)
#   v-point  at (i,   j+½)  —  meridional velocity v  (staggered ½-cell in y)
#   ζ-point  at (i+½, j+½)  —  vorticity ζ            (staggered ½-cell in both)
#
# avg_x / delta_x shift the x-position by ½:   p ↔ u,  v ↔ ζ
# avg_y / delta_y shift the y-position by ½:   p ↔ v,  u ↔ ζ
#
# Each function reduces shape by 1 in its operating dimension:
#   avg_x(f) / delta_x(dx, f)  : (Mx, My) → (Mx-1, My)
#   avg_y(f) / delta_y(dy, f)  : (Mx, My) → (Mx, My-1)
#
# For an (M+2, N+2) halo array the interior of any point type is at [1:M+1, 1:N+1].
# After an x-stencil op the (M+1, …) result's interior slice depends on input type:
#   p- or v-point input  →  u- or ζ-point output,  interior at [1:M+1, ...]
#   u- or ζ-point input  →  p- or v-point output,  interior at [0:M,   ...]
# After a y-stencil op:
#   p- or u-point input  →  v- or ζ-point output,  interior at [..., 1:N+1]
#   v- or ζ-point input  →  p- or u-point output,  interior at [..., 0:N  ]
# ---------------------------------------------------------------------------


def avg_x(f):
    """avg_x(f)[i,j] = 0.5*(f[i+1,j] + f[i,j])  — reduces x by 1"""
    return 0.5 * (f[1:, :] + f[:-1, :])


def avg_y(f):
    """avg_y(f)[i,j] = 0.5*(f[i,j+1] + f[i,j])  — reduces y by 1"""
    return 0.5 * (f[:, 1:] + f[:, :-1])


def delta_x(dx, f):
    """delta_x(f)[i,j] = (1/dx)*(f[i+1,j] - f[i,j])  — reduces x by 1"""
    return (1.0 / dx) * (f[1:, :] - f[:-1, :])


def delta_y(dy, f):
    """delta_y(f)[i,j] = (1/dy)*(f[i,j+1] - f[i,j])  — reduces y by 1"""
    return (1.0 / dy) * (f[:, 1:] - f[:, :-1])


def timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N):
    """Perform one timestep of the shallow water equations.

    All fields have shape (M+2, N+2) with 1-wide symmetric halos.
    Implements the Sadourny (1975) Arakawa C-grid scheme via direct array composition.
    Grid-point types follow Sadourny's notation  (p-, u-, v-, ζ-points):

      cu  = avg_x(p) * u          # u-point  (i+½, j)
      cv  = avg_y(p) * v          # v-point  (i, j+½)
      z   = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))  # ζ-point  (i+½, j+½)
      h   = p + 0.5*(avg_x(u²) + avg_y(v²))              # p-point  (i, j)

      unew  at u-point:  avg_y(z) [ζ→u] * avg_y(avg_x(cv)) [v→ζ→u] * dt
                       − delta_x(h)     [p→u]                       * dt
      vnew  at v-point: −avg_x(z) [ζ→v] * avg_x(avg_y(cu)) [u→ζ→v] * dt
                       − delta_y(h)     [p→v]                       * dt
      pnew  at p-point: −delta_x(cu)    [u→p]                       * dt
                       − delta_y(cv)    [v→p]                       * dt

    Each stencil function reduces shape by 1 only in its operating dimension, so
    compositions chain directly without any intermediate _interior_to_halo calls.

    Slice convention for extracting the (M, N) interior from a composed result
    (see stencil-block header for the full rule):
      p- or v-point input → x-result at u- or ζ-point: interior rows [1:M+1, ...]
      u- or ζ-point input → x-result at p- or v-point: interior rows [0:M,   ...]
      p- or u-point input → y-result at v- or ζ-point: interior cols [..., 1:N+1]
      v- or ζ-point input → y-result at p- or u-point: interior cols [..., 0:N  ]
    """
    uu = u * u
    vv = v * v
    # cu at u-points: avg_x(p) [p→u] * u[0:M+1,:]  → shape (M+1, N+2)
    cu = avg_x(p) * u[:-1, :]
    # cv at v-points: avg_y(p) [p→v] * v[:,0:N+1]  → shape (M+2, N+1)
    cv = avg_y(p) * v[:, :-1]

    # z_wide at ζ-points (i+½, j+½), shape (M+1, N+1):
    #   delta_x(v) [v→ζ]: shape (M+1, N+2); [:, :N+1] trims to (M+1, N+1)
    #   delta_y(u) [u→ζ]: shape (M+2, N+1); [:M+1, :] trims to (M+1, N+1)
    #   avg_x(avg_y(p)) [p→v→ζ]: shape (M+1, N+1)
    z_wide = (delta_x(dx, v)[:, : N + 1] - delta_y(dy, u)[: M + 1, :]) / avg_x(
        avg_y(p)
    )  # (M+1, N+1), ζ-points

    # h_wide at p-points covering x=1..M+1, y=1..N+1 (interior + right/top halo),
    # shape (M+1, N+1).  The extended range is needed by the subsequent delta_x/delta_y.
    #   p[1:M+2, 1:N+2]            : p-points at x=1..M+1, y=1..N+1
    #   avg_x(uu) [u→p]: result[k] at x=k+1; slice [0:M+1, 1:N+2] → x=1..M+1
    #   avg_y(vv) [v→p]: result[l] at y=l+1; slice [1:M+2, 0:N+1] → y=1..N+1
    h_wide = p[1 : M + 2, 1 : N + 2] + 0.5 * (
        avg_x(uu)[0 : M + 1, 1 : N + 2] + avg_y(vv)[1 : M + 2, 0 : N + 1]
    )  # (M+1, N+1), p-points

    # unew at interior u-points: all three terms evaluated at (i+½, j) for i=1..M, j=1..N.
    #   avg_y(z_wide)    [ζ→u]: z_wide ζ-point → avg_y → u-point; interior at [1:M+1, 0:N]
    #   avg_y(avg_x(cv)) [v→ζ→u]: cv v-point → avg_x →ζ-point → avg_y → u-point; same slice
    #   delta_x(h_wide)  [p→u]: h_wide p-point at x=k+1 → delta_x → u-point at x=k+3/2;
    #                           interior u at x=3/2..M+1/2 ↔ slice [0:M, 0:N]
    unew_interior = (
        uold
        + avg_y(z_wide)[1 : M + 1, 0:N] * avg_y(avg_x(cv))[1 : M + 1, 0:N] * dt_val
        - delta_x(dx, h_wide)[0:M, 0:N] * dt_val
    )
    # vnew at interior v-points: all three terms evaluated at (i, j+½) for i=1..M, j=1..N.
    #   avg_x(z_wide)    [ζ→v]: z_wide ζ-point → avg_x → v-point; interior at [0:M, 1:N+1]
    #   avg_x(avg_y(cu)) [u→ζ→v]: cu u-point → avg_y → ζ-point → avg_x → v-point; same slice
    #   delta_y(h_wide)  [p→v]: h_wide p-point at y=l+1 → delta_y → v-point at y=l+3/2;
    #                           interior v at y=3/2..N+1/2 ↔ slice [0:M, 0:N]
    vnew_interior = (
        vold
        - avg_x(z_wide)[0:M, 1 : N + 1] * avg_x(avg_y(cu))[0:M, 1 : N + 1] * dt_val
        - delta_y(dy, h_wide)[0:M, 0:N] * dt_val
    )
    # pnew at interior p-points: both terms evaluated at (i, j) for i=1..M, j=1..N.
    #   delta_x(cu) [u→p]: cu u-point → delta_x → p-point; interior at [0:M, 1:N+1]
    #   delta_y(cv) [v→p]: cv v-point → delta_y → p-point; interior at [1:M+1, 0:N]
    pnew_interior = (
        pold - delta_x(dx, cu)[0:M, 1 : N + 1] * dt_val - delta_y(dy, cv)[1 : M + 1, 0:N] * dt_val
    )

    # -- Time filter (update old fields) --
    uold_new = u[1 : M + 1, 1 : N + 1] + alpha_val * (
        unew_interior - 2.0 * u[1 : M + 1, 1 : N + 1] + uold
    )
    vold_new = v[1 : M + 1, 1 : N + 1] + alpha_val * (
        vnew_interior - 2.0 * v[1 : M + 1, 1 : N + 1] + vold
    )
    pold_new = p[1 : M + 1, 1 : N + 1] + alpha_val * (
        pnew_interior - 2.0 * p[1 : M + 1, 1 : N + 1] + pold
    )

    # Build full arrays with halos for all returned fields.
    unew = _interior_to_halo(xp, unew_interior)
    vnew = _interior_to_halo(xp, vnew_interior)
    pnew = _interior_to_halo(xp, pnew_interior)

    return unew, vnew, pnew, uold_new, vold_new, pold_new


def main():
    parser = argparse.ArgumentParser(description="Shallow Water Model (Array API)")
    parser.add_argument(
        "--array-library",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "torch", "cupy", "array_api_strict"],
        help="Array library to use",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable array-api-strict compliance checking via array_api_compat",
    )
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--ITMAX", type=int, default=4000)
    parser.add_argument("--validate", action="store_true", help="Validate against reference data")
    parser.add_argument("--validate-deep", action="store_true", help="Deep validation of each step")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable JIT compilation (jax.jit for jax, torch.compile for torch)",
    )
    parser.add_argument("--no-output", action="store_true", help="Suppress diagnostic output")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run on (default: CPU for numpy/torch, GPU for jax if available)",
    )
    args = parser.parse_args()

    M = args.M
    N = args.N
    ITMAX = args.ITMAX
    L_OUT = not args.no_output

    # Physical parameters
    dt = 90.0
    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    alpha = 0.001

    # Get the array module
    if args.array_library == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        if args.device is not None:
            jax_device_kind = "gpu" if args.device == "cuda" else "cpu"
            jax.config.update("jax_default_device", jax.devices(jax_device_kind)[0])
            print(f"JAX device: {jax_device_kind}")

    lib = _get_array_module(args.array_library)

    # Configure torch device
    if args.array_library == "torch" and args.device is not None:
        import torch

        torch.set_default_device(args.device)
        print(f"Torch device: {args.device}")
    elif args.device == "cuda" and args.array_library not in ("jax", "torch", "cupy"):
        print(f"Warning: --device cuda not supported for {args.array_library}")

    if args.strict:
        import array_api_strict

        array_api_strict.set_array_api_strict_flags(api_version="2024.12")

    if args.strict and args.array_library != "array_api_strict":
        # Wrap the library arrays in array_api_strict for compliance testing
        import array_api_strict

        xp = array_api_strict
        print(
            f"Running with {args.array_library} arrays wrapped in array_api_strict for compliance checking"
        )
    elif args.strict:
        import array_api_strict

        xp = array_api_strict
        print("Running with array_api_strict directly")
    else:
        xp = lib
        # Get the array-api-compat namespace for the library
        test_arr = lib.zeros((1,))
        xp = array_namespace(test_arr)
        print(f"Running with {args.array_library} via array_api_compat namespace")

    # Initialize fields
    u, v, p = initial_conditions.initialize_2halo(xp, M, N, dx, dy, a)

    if args.strict and args.array_library != "array_api_strict":
        # Convert to array_api_strict arrays
        import array_api_strict
        import numpy as np

        u_np = np.asarray(u) if hasattr(u, "__array__") else u
        v_np = np.asarray(v) if hasattr(v, "__array__") else v
        p_np = np.asarray(p) if hasattr(p, "__array__") else p
        u = array_api_strict.asarray(u_np, dtype=array_api_strict.float64)
        v = array_api_strict.asarray(v_np, dtype=array_api_strict.float64)
        p = array_api_strict.asarray(p_np, dtype=array_api_strict.float64)
        xp = array_api_strict

    uold = xp.asarray(u, copy=True)[1 : M + 1, 1 : N + 1]
    vold = xp.asarray(v, copy=True)[1 : M + 1, 1 : N + 1]
    pold = xp.asarray(p, copy=True)[1 : M + 1, 1 : N + 1]

    if L_OUT:
        print(f" Number of points in the x direction: {M}")
        print(f" Number of points in the y direction: {N}")
        print(f" grid spacing in the x direction: {dx}")
        print(f" grid spacing in the y direction: {dy}")
        print(f" time step: {dt}")
        print(f" time filter coefficient: {alpha}")

    # For validation, we need numpy
    if args.validate or args.validate_deep:
        import numpy as np

    if args.validate_deep:
        import sys

        sys.path.insert(0, "/home/user/SWM/swm_python")
        import utils

    # Set up the timestep function, optionally with JIT compilation
    def timestep_fn(u, v, p, uold, vold, pold, dt_val, alpha_val):
        return timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N)

    if args.compile:
        if args.array_library == "jax":
            import jax

            timestep_fn = jax.jit(timestep_fn)
            print("JIT compilation enabled via jax.jit")
        elif args.array_library == "torch":
            import torch

            timestep_fn = torch.compile(timestep_fn)
            print("JIT compilation enabled via torch.compile")
        else:
            print(f"Warning: --compile has no effect for {args.array_library}")

        # Warm-up call to trigger compilation before timing
        print("Warm-up call...", end=" ", flush=True)
        t_warmup_start = perf_counter()
        _warmup_result = timestep_fn(u, v, p, uold, vold, pold, dt, 0.0)
        if args.array_library == "jax":
            _warmup_result[0].block_until_ready()
        elif args.array_library == "torch" and args.device == "cuda":
            torch.cuda.synchronize()
        t_warmup_stop = perf_counter()
        del _warmup_result
        print(f"done ({t_warmup_stop - t_warmup_start:.3f}s)")

    dt_total = 0.0
    dt_compute = 0.0

    t0_start = perf_counter()

    for ncycle in range(ITMAX):
        if ncycle % 100 == 0:
            print(f"cycle number {ncycle}")

        if args.validate_deep and ncycle <= 3:
            import numpy as np

            u_np = _to_numpy(u)
            v_np = _to_numpy(v)
            p_np = _to_numpy(p)
            # Convert 2-halo to reference layout
            utils.validate_uvp(
                u_np[:-1, 1:],
                v_np[1:, :-1],
                p_np[1:, 1:],
                M,
                N,
                ncycle,
                "init",
            )

        tdt = dt if ncycle == 0 else dt * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        t_start = perf_counter()
        unew, vnew, pnew, uold, vold, pold = timestep_fn(u, v, p, uold, vold, pold, tdt, alpha_val)
        t_stop = perf_counter()
        dt_compute += t_stop - t_start

        u = unew
        v = vnew
        p = pnew

    # Synchronize device for accurate timing
    if args.array_library == "jax":
        u.block_until_ready()
    elif args.array_library == "torch" and args.device == "cuda":
        import torch

        torch.cuda.synchronize()

    t0_stop = perf_counter()
    dt_total = t0_stop - t0_start

    if L_OUT:
        print(f"cycle number {ITMAX}")

    print(f"total: {dt_total}")
    print(f"compute: {dt_compute}")

    if args.validate:
        import numpy as np

        u_np = _to_numpy(u)
        v_np = _to_numpy(v)
        p_np = _to_numpy(p)

        # Convert to reference layout for validation
        u_ref_layout = u_np[:-1, 1:]
        v_ref_layout = v_np[1:, :-1]
        p_ref_layout = p_np[1:, 1:]

        sys_path_added = False
        import sys

        if "/home/user/SWM/swm_python" not in sys.path:
            sys.path.insert(0, "/home/user/SWM/swm_python")
            sys_path_added = True
        import utils

        utils.final_validation(u_ref_layout, v_ref_layout, p_ref_layout, ITMAX=ITMAX, M=M, N=N)


if __name__ == "__main__":
    main()
