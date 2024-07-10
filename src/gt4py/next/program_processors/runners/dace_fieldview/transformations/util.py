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

"""Common functionality for the transformations."""

import dace


def is_nested_sdfg(sdfg: dace.SDFG) -> bool:
    """Tests if `sdfg` is a neseted sdfg."""
    if isinstance(sdfg, dace.SDFGState):
        sdfg = sdfg.parent
    if isinstance(sdfg, dace.nodes.NestedSDFG):
        return True
    elif isinstance(sdfg, dace.SDFG):
        if sdfg.parent_nsdfg_node is not None:
            return True
        return False
    else:
        raise TypeError(f"Does not know how to handle '{type(sdfg).__name__}'.")
