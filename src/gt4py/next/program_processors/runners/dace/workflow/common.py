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
    """Set the DaCe configuration as required by GT4Py.

    This function acts on the current configuration and should not be used inside
    a context, such as `dace.config.temporary_config()`, see note below for more.

    Args:
        device_type: Target device type, needed for compiler config.
        cmake_build_type: CMake build type, needed for compiler config.

    Note:
        Because of the reasons described in [DaCe issue#2125](https://github.com/spcl/dace/issues/2125)
        it is not safe to use this function inside a `dace.config.temporary_config()`
        context or use it to set options that are not static for the process.
    """
    # NOTE: As explained in [DaCe issue#2125](https://github.com/spcl/dace/issues/2125)
    #   it is not possible to use this function inside a configuration context,
    #   or use it to set anything that is not specific to the process. As an example
    #   we can configure which GPU backend is used, under the assumption that we will
    #   only use one type of GPU. Furthermore, we have to set always every time, i.e.
    #   we have to set all flags for the GPU backend even if we are using CPU, since
    #   another thread might work with the GPU right now.

    # We rely on dace cache to avoid recompiling the SDFG.
    #   Note that the workflow step with the persistent `FileCache` store
    #   is translating from `CompilableProgram` (ITIR.Program + CompileTimeArgs)
    #   to `ProgramSource`, so this step is storing in cache only the result
    #   of the SDFG transformations, not the compiled program binary.
    dace.Config.set("cache", value="hash")  # use the SDFG hash as cache key
    dace.Config.set("default_build_folder", value=str(config.BUILD_CACHE_DIR / "dace_cache"))
    dace.Config.set("compiler.use_cache", value=True)

    # Prevents the implicit change of Memlets to Maps. Instead they should be handled by
    #  `gt4py.next.program_processors.runners.dace.transfromations.gpu_utils.gt_gpu_transform_non_standard_memlet()`.
    dace.Config.set("compiler.cuda.allow_implicit_memlet_to_map", value=False)

    if cmake_build_type is not None:
        dace.Config.set("compiler.build_type", value=cmake_build_type.value)

    # dace dafault setting use fast-math in both cpu and gpu compilation, don't use it here
    dace.Config.set(
        "compiler.cpu.args",
        value="-std=c++14 -fPIC -O3 -march=native -Wall -Wextra -Wno-unused-parameter -Wno-unused-label",
    )
    dace.Config.set(
        "compiler.cuda.args",
        value="-Xcompiler -O3 -Xcompiler -march=native -Xcompiler -Wno-unused-parameter",
    )
    dace.Config.set(
        "compiler.cuda.hip_args",
        value="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter",
    )

    # In some stencils, for example `apply_diffusion_to_w`, the cuda codegen messes
    #  up with the cuda streams, i.e. it allocates N streams but uses N+1. The first
    #  idea was to use just one stream. However, even in that case the generator
    #  generated wrong code. The current approach is to use the default stream, i.e.
    #  setting `max_concurrent_streams` to `-1`. However, the draw back is, that
    #  apparently then all synchronization is disabled, even the one at the very
    #  end of the SDFG call. To correct for that we are using either
    #  `make_sdfg_call_sync()` or `make_sdfg_call_async()`, see there or in
    #  [DaCe issue#2120](https://github.com/spcl/dace/issues/2120) for more.
    dace.Config.set("compiler.cuda.max_concurrent_streams", value=-1)

    # This assumes that a process will only use one type of GPU.
    if device_type == core_defs.DeviceType.ROCM:
        dace.Config.set("compiler.cuda.backend", value="hip")
    elif device_type == core_defs.DeviceType.CUDA:
        dace.Config.set("compiler.cuda.backend", value="cuda")

    # Instrumentation of SDFG timers
    dace.Config.set("instrumentation", "report_each_invocation", value=True)

    # we are not interested in storing the history of SDFG transformations.
    dace.Config.set("store_history", value=False)
