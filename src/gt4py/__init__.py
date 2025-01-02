# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Python library for generating high-performance implementations of stencil kernels for weather and climate modeling."""

import sys as _sys

from . import cartesian, eve, storage
from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
    "cartesian",
    "eve",
    "storage",
]


if _sys.version_info >= (3, 10):
    from . import next  # noqa: A004 shadowing a Python builtin

    __all__ += ["next"]
