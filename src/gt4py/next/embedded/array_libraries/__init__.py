# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` of the array libraries whose arrays need more than the Array API standard, registered if importable (torch: when its first tensor is seen)."""

import importlib

from gt4py.next.embedded import array_ops

from . import numpy_ops as numpy_ops


for _library in ("cupy", "jax"):
    try:
        importlib.import_module(_library)
    except ImportError:
        continue
    importlib.import_module(f"{__name__}.{_library}_ops")

# importing torch takes about a second
array_ops.register_on_first_use("array_api_compat.torch", f"{__name__}.torch_ops")
