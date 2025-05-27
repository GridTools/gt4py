# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import config


def set_dace_config(
    device_type: core_defs.DeviceType,
    cmake_build_type: Optional[config.CMakeBuildType] = None,
) -> None:
    """Set dace configuration, shared among all workflow stages.

    Args:
        device_type: Target device type, needed for compiler config.
        cmake_build_type: CMake build type, needed for compiler config.
    """

    # We rely on dace cache to avoid recompiling the SDFG.
    #   Note that the workflow step with the persistent `FileCache` store
    #   is translating from `CompilableProgram` (ITIR.Program + CompileTimeArgs)
    #   to `ProgramSource`, so this step is storing in cache only the result
    #   of the SDFG transformations, not the compiled program binary.
    dace.config.Config.set("cache", value="hash")  # use the SDFG hash as cache key
    dace.config.Config.set("default_build_folder", value=str(config.BUILD_CACHE_DIR / "dace_cache"))
    dace.config.Config.set("compiler.use_cache", value=True)

    if cmake_build_type is not None:
        dace.config.Config.set("compiler.build_type", value=cmake_build_type.value)

    # dace dafault setting use fast-math in both cpu and gpu compilation, don't use it here
    dace.config.Config.set(
        "compiler.cpu.args",
        value="-std=c++14 -fPIC -O3 -march=native -Wall -Wextra -Wno-unused-parameter -Wno-unused-label",
    )
    dace.config.Config.set(
        "compiler.cuda.args",
        value="-Xcompiler -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter",
    )
    dace.config.Config.set(
        "compiler.cuda.hip_args",
        value="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter",
    )

    # In some stencils, mostly in `apply_diffusion_to_w` the cuda codegen messes
    #  up with the cuda streams, i.e. it allocates N streams but uses N+1.
    #  This is a workaround until this issue if fixed in DaCe.
    dace.config.Config.set("compiler.cuda.max_concurrent_streams", value=1)

    if device_type == core_defs.DeviceType.ROCM:
        dace.config.Config.set("compiler.cuda.backend", value="hip")

    # Instrumentation of SDFG timers
    dace.config.Config.set("instrumentation", "report_each_invocation", value=True)
