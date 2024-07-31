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


from gt4py.next.program_processors.runners.dace_fieldview.gtir_dace_backend import get_sdfg_args
from gt4py.next.program_processors.runners.dace_fieldview.gtir_to_sdfg import build_sdfg_from_gtir


__all__ = [
    "build_sdfg_from_gtir",
    "get_sdfg_args",
]
