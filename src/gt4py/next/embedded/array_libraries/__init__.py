# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`ArrayOps` of the array libraries whose arrays need more than the Array API standard, registered if installed."""

import importlib
import importlib.util

from . import numpy_ops as numpy_ops


for _library in ("cupy", "jax"):
    if importlib.util.find_spec(_library) is not None:
        importlib.import_module(f"{__name__}.{_library}_ops")
