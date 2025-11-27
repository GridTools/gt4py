# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from warnings import warn

from .base import REGISTRY, Backend, BaseBackend, BasePyExtBackend, from_name, register
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
    "DebugBackend",
    "GTCpuIfirstBackend",
    "GTCpuKfirstBackend",
    "GTGpuBackend",
    "NumpyBackend",
    "from_name",
    "register",
]


try:
    from .dace_backend import DaceCPUBackend, DaceCPUKFirstBackend, DaceGPUBackend

    __all__ += ["DaceCPUBackend", "DaceCPUKFirstBackend", "DaceGPUBackend"]
except ImportError:
    warn(
        "GT4Py was unable to load DaCe. DaCe backends (`dace:cpu`, `dace:cpu_kfirst`, and `dace:gpu`) will not be available.",
        stacklevel=2,
    )
