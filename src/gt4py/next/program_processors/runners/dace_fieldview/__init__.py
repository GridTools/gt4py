# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.next.program_processors.runners.dace_common.dace_backend import get_sdfg_args
from gt4py.next.program_processors.runners.dace_fieldview.gtir_to_sdfg import build_sdfg_from_gtir


__all__ = [
    "build_sdfg_from_gtir",
    "get_sdfg_args",
]
