# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of GTScript: an embedded DSL in Python for stencil computations.

Interface functions to define and compile GTScript definitions and empty symbol
definitions for the keywords of the DSL.
"""

import sys

from gt4py.cartesian.gtscript import *  # noqa: F403 [undefined-local-with-import-star]


sys.modules["__gtscript__"] = sys.modules["gt4py.cartesian.__gtscript__"]
