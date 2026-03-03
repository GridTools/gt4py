# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Shallow Water Model using the Python Array API standard.

This implementation uses symmetric halo lines (1 on each side) for all fields,
following the approach of swm_next2_halo2_restructured.py. The periodic boundary
conditions are applied via halo exchange rather than asymmetric padding.

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
# Stencil building blocks – mirror the @gtx.field_operator functions in swm.py.
#
# Each function reduces shape by 1 only in its operating dimension, preserving
# the full extent in the other dimension.  No M/N parameters are needed:
#
#   avg_x(f)   / delta_x(dx, f)   : (..., Mx, My) → (..., Mx-1, My)
#   avg_y(f)   / delta_y(dy, f)   : (..., Mx, My) → (..., Mx, My-1)
#
# Both "forward" (avg_x) and "staggered/backward" (avg_x_staggered) averages
# produce the same array 0.5*(f[1:]+f[:-1]).  The semantic difference is which
# output slice maps to the interior — callers select the appropriate rows/cols:
#   forward  in x → interior at rows [1:M+1, ...]   (index i maps to position i)
#   backward in x → interior at rows [0:M,   ...]   (index i maps to position i+1)
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
    Mirrors the @gtx.field_operator timestep in swm.py via direct composition:

      cu = avg_x(p) * u            # (M+1, N+2)
      cv = avg_y(p) * v            # (M+2, N+1)
      z  = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))   # (M+1, N+1)
      h  = p + 0.5*(avg_x_staggered(u*u) + avg_y_staggered(v*v))  # (M+1, N+1)

      unew = uold + avg_y_staggered(z)*avg_y_staggered(avg_x(cv))*dt - delta_x(h)*dt
      vnew = vold - avg_x_staggered(z)*avg_x_staggered(avg_y(cu))*dt - delta_y(h)*dt
      pnew = pold - delta_x_staggered(cu)*dt - delta_y_staggered(cv)*dt

    Each stencil function reduces shape by 1 only in its operating dimension, so
    compositions chain directly without any intermediate _interior_to_halo calls.

    Slice convention for extracting the (M, N) interior from composed results:
      forward  avg/delta in x → rows [1:M+1, ...]   (index i means position i)
      backward avg/delta in x → rows [0:M,   ...]   (index i means position i+1)
      forward  avg/delta in y → cols [..., 1:N+1]
      backward avg/delta in y → cols [..., 0:N  ]
    """
    uu = u * u
    vv = v * v
    # cu[i,j] = 0.5*(p[i+1,j]+p[i,j]) * u[i,j]  for i=0..M, j=0..N+1  → (M+1, N+2)
    cu = avg_x(p) * u[:-1, :]
    # cv[i,j] = 0.5*(p[i,j+1]+p[i,j]) * v[i,j]  for i=0..M+1, j=0..N  → (M+2, N+1)
    cv = avg_y(p) * v[:, :-1]

    # z_wide at positions i=0..M, j=0..N (interior + 1 halo needed for staggered avgs)
    # delta_x(v) → (M+1,N+2), slice [:,:N+1] keeps j=0..N
    # delta_y(u) → (M+2,N+1), slice [:M+1,:] keeps i=0..M
    # avg_x(avg_y(p)) → (M+1,N+1) exactly
    z_wide = (delta_x(dx, v)[:, : N + 1] - delta_y(dy, u)[: M + 1, :]) / avg_x(
        avg_y(p)
    )  # (M+1, N+1)

    # h_wide at positions i=1..M+1, j=1..N+1 (interior + 1 extra for forward delta)
    #   avg_x_staggered(uu)[i,j] = avg_x(uu)[i-1,j]  → avg_x(uu)[0:M+1, 1:N+2]
    #   avg_y_staggered(vv)[i,j] = avg_y(vv)[i,j-1]  → avg_y(vv)[1:M+2, 0:N+1]
    h_wide = p[1 : M + 2, 1 : N + 2] + 0.5 * (
        avg_x(uu)[0 : M + 1, 1 : N + 2] + avg_y(vv)[1 : M + 2, 0 : N + 1]
    )  # (M+1, N+1)

    # avg_y_staggered(z)[i,j] = avg_y(z_wide)[i, j-1]  → [1:M+1, 0:N]
    # avg_y_staggered(avg_x(cv))[i,j] = avg_y(avg_x(cv))[i, j-1]  → [1:M+1, 0:N]
    # delta_x(h)[i,j] = delta_x(h_wide)[i-1, j-1]  → [0:M, 0:N]
    unew_interior = (
        uold
        + avg_y(z_wide)[1 : M + 1, 0:N] * avg_y(avg_x(cv))[1 : M + 1, 0:N] * dt_val
        - delta_x(dx, h_wide)[0:M, 0:N] * dt_val
    )
    # avg_x_staggered(z)[i,j] = avg_x(z_wide)[i-1, j]  → [0:M, 1:N+1]
    # avg_x_staggered(avg_y(cu))[i,j] = avg_x(avg_y(cu))[i-1, j]  → [0:M, 1:N+1]
    # delta_y(h)[i,j] = delta_y(h_wide)[i-1, j-1]  → [0:M, 0:N]
    vnew_interior = (
        vold
        - avg_x(z_wide)[0:M, 1 : N + 1] * avg_x(avg_y(cu))[0:M, 1 : N + 1] * dt_val
        - delta_y(dy, h_wide)[0:M, 0:N] * dt_val
    )
    # delta_x_staggered(cu)[i,j] = delta_x(cu)[i-1, j]  → [0:M, 1:N+1]
    # delta_y_staggered(cv)[i,j] = delta_y(cv)[i, j-1]  → [1:M+1, 0:N]
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


def to_reference_layout(arr, M, N):
    """Convert from 2-halo (M+2, N+2) layout to reference (M+1, N+1) layout.

    The reference data uses an asymmetric layout where:
      u: padded with (1,0) in x and (0,1) in y  -> u_ref = [halo; interior_rows][interior_cols; halo]
      v: padded with (0,1) in x and (1,0) in y
      p: padded with (0,1) in x and (0,1) in y

    For the 2-halo symmetric layout, the interior is at [1:M+1, 1:N+1].
    The reference format stores M+1 x N+1 values.

    For u: ref has rows [M, 0..M-1] and cols [0..N-1, 0] -> u_ref = u_2halo[0:M+1, 1:N+2]
      which is u_2halo[:-1, 1:]
    For v: ref has rows [0..N-1, 0] and cols [N, 0..N-1] -> v_ref = u_2halo[1:M+2, 0:N+1]
      which is v_2halo[1:, :-1]
    For p: ref has rows [0..M-1, 0] and cols [0..N-1, 0] -> p_ref = p_2halo[1:M+2, 1:N+2]
      which is p_2halo[1:, 1:]
    """
    pass  # implemented inline in validation


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
