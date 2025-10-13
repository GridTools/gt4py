# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
GT4Py-NEXT - Performance portable and composable weather & climate stencils.

This module deviates from the project coding style (Google Python style) in the following way:

[style guide link](https://google.github.io/styleguide/pyguide.html#22-imports):

Fnctions and classes are imported from other modules in order to explicitly re-export them
to create a streamlined user experience. `from module import *` can be used but only if the
module in question is a submodule, defines `__all__` and exports many public API objects.
"""

# ruff: noqa: F401
from .._core.definitions import CUPY_DEVICE_TYPE, Device, DeviceType, is_scalar_type
from . import common, ffront, iterator, program_processors, typing
from .common import (
    Connectivity,
    Dimension,
    DimensionKind,
    Dims,
    Domain,
    Field,
    GridType,
    UnitRange,
    domain,
    unit_range,
)
from .constructors import as_connectivity, as_field, empty, full, ones, zeros
from .embedded import (  # Just for registering field implementations
    nd_array_field as _nd_array_field,
)
from .ffront import fbuiltins
from .ffront.decorator import field_operator, program, scan_operator
from .ffront.fbuiltins import *  # noqa: F403 [undefined-local-with-import-star]  explicitly reexport all from fbuiltins.__all__
from .ffront.fbuiltins import FieldOffset
from .otf.compiled_program import wait_for_compilation
from .program_processors.runners.gtfn import (
    run_gtfn_cached as gtfn_cpu,
    run_gtfn_gpu_cached as gtfn_gpu,
)
from .program_processors.runners.roundtrip import default as itir_python


__all__ = [
    # submodules
    "common",
    "ffront",
    "iterator",
    "program_processors",
    "typing",
    # from _core.definitions
    "CUPY_DEVICE_TYPE",
    "Device",
    "DeviceType",
    "is_scalar_type",
    # from common
    "Dimension",
    "DimensionKind",
    "Dims",
    "Field",
    "Connectivity",
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
    # from ffront
    "FieldOffset",
    "field_operator",
    "program",
    "scan_operator",
    # from otf
    "wait_for_compilation",
    # from program_processor
    "gtfn_cpu",
    "gtfn_gpu",
    "itir_python",
    *fbuiltins.__all__,
]
