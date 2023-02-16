# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
    from . import next

    __all__ += ["next"]
