# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

"""Implementation of GTScript: an embedded DSL in Python for stencil computations.

Interface functions to define and compile GTScript definitions and empty symbol
definitions for the keywords of the DSL.
"""

import sys

from gt4py.cartesian.gtscript import *  # noqa: F403 [undefined-local-with-import-star]


sys.modules["__gtscript__"] = sys.modules["gt4py.cartesian.__gtscript__"]
