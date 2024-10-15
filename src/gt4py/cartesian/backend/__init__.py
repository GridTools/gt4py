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

from .base import (
    REGISTRY,
    Backend,
    BaseBackend,
    BasePyExtBackend,
    CLIBackendMixin,
    PurePythonBackendCLIMixin,
    from_name,
    register,
)


try:
    from .dace_backend import DaceCPUBackend, DaceGPUBackend
except ImportError:
    pass

from .cuda_backend import CudaBackend
from .debug_backend import DebugBackend
from .gtcpp_backend import GTCpuIfirstBackend, GTCpuKfirstBackend, GTGpuBackend
from .module_generator import BaseModuleGenerator
from .numpy_backend import NumpyBackend


__all__ = [
    "REGISTRY",
    "Backend",
    "BaseBackend",
    "BaseModuleGenerator",
    "BasePyExtBackend",
    "CLIBackendMixin",
    "CudaBackend",
    "DebugBackend",
    "GTGpuBackend",
    "GTCpuIfirstBackend",
    "GTCpuKfirstBackend",
    "NumpyBackend",
    "PurePythonBackendCLIMixin",
    "from_name",
    "register",
]


if "DaceCPUBackend" in globals():
    __all__ += ["DaceCPUBackend", "DaceGPUBackend"]
