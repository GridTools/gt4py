# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.backend.base import REGISTRY as BACKEND_REGISTRY


# TODO (romanc)
# This file can move to the tests/ folder once we refactor the `StencilTestSuite`
# class (i.e. https://github.com/GEOS-ESM/NDSL/issues/72). The stencil test suites
# use the `GPU_BACKEND_NAMES` and I didn't wanna have `gt4py/cartesian/testing` depend
# on the `tests/` directory. That sounded the wrong way around, so I moved them here
# for now.

ALL_BACKEND_NAMES = list(BACKEND_REGISTRY.keys())

GPU_BACKEND_NAMES = ["cuda", "gt:gpu", "dace:gpu"]
CPU_BACKEND_NAMES = [name for name in ALL_BACKEND_NAMES if name not in GPU_BACKEND_NAMES]

PERFORMANCE_BACKEND_NAMES = [name for name in ALL_BACKEND_NAMES if name not in ("numpy", "cuda")]
