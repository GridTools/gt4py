# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Python library for generating high-performance implementations of stencil kernels for weather and climate modeling.

The library carries to implementations available as submodule imports:
    - `gt4py.cartesian` is a cartesian grid restricted,
    - `gt4py.next` supports structured and unstructured grid.
"""

from . import eve, storage
from .__about__ import __author__, __copyright__, __license__, __version__, __version_info__


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
    "eve",
    "storage",
]
