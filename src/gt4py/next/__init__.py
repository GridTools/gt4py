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
"""
GT4Py-NEXT - Performance portable and composable weather & climate stencils.

This module deviates from the project coding style (Google Python style) in the following way:

[style guide link](https://google.github.io/styleguide/pyguide.html#22-imports):

Fnctions and classes are imported from other modules in order to explicitly re-export them
to create a streamlined user experience. `from module import *` can be used but only if the
module in question is a submodule, defines `__all__` and exports many public API objects.
"""

from . import common, ffront, iterator, program_processors, type_inference
from .common import Dimension, DimensionKind, Domain, Field, GridType, UnitRange, domain, unit_range
from .constructors import as_connectivity, as_field, empty, full, ones, zeros
from .embedded import (  # Just for registering field implementations
    nd_array_field as _nd_array_field,
)
from .ffront import fbuiltins
from .ffront.decorator import field_operator, program, scan_operator
from .ffront.fbuiltins import *  # noqa: F403 [undefined-local-with-import-star]  explicitly reexport all from fbuiltins.__all__
from .ffront.fbuiltins import FieldOffset
from .iterator.embedded import (
    NeighborTableOffsetProvider,
    StridedNeighborOffsetProvider,
    index_field,
    np_as_located_field,
)
from .program_processors.runners.gtfn import (
    run_gtfn_cached as gtfn_cpu,
    run_gtfn_gpu_cached as gtfn_gpu,
)
from .program_processors.runners.roundtrip import backend as itir_python


__all__ = [
    # submodules
    "common",
    "ffront",
    "iterator",
    "program_processors",
    "type_inference",
    # from common
    "Dimension",
    "DimensionKind",
    "Field",
    "GridType",
    "domain",
    "Domain",
    "unit_range",
    "UnitRange",
    # from constructors
    "empty",
    "zeros",
    "ones",
    "full",
    "as_field",
    "as_connectivity",
    # from iterator
    "NeighborTableOffsetProvider",
    "StridedNeighborOffsetProvider",
    "index_field",
    "np_as_located_field",
    # from ffront
    "FieldOffset",
    "field_operator",
    "program",
    "scan_operator",
    # from program_processor
    "gtfn_cpu",
    "gtfn_gpu",
    "itir_python",
    *fbuiltins.__all__,
]
