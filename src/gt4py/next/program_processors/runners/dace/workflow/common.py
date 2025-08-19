# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from typing import Any, Generator, Optional

import dace

from gt4py._core import definitions as core_defs
from gt4py.next import config


def set_dace_config(
    device_type: core_defs.DeviceType,
    cmake_build_type: Optional[config.CMakeBuildType] = None,
) -> None:
    """Set the DaCe configuration as required by GT4Py.

    This function should never be used directly, instead the context manager
    `dace_context()` should be used!
    Furthermore, all changes to `dace.Config` are applied by this function.

    Args:
        device_type: Target device type, needed for compiler config.
        cmake_build_type: CMake build type, needed for compiler config.

    Note:
        For every thread DaCe will maintain a separate set of configuration. Thus,
        the will not influence each other. It is also important that a thread will
        not inherent the configuration of its parent, but it will be initialized
        to the default. This means that this function should be always called.
        When working on DaCe.
    """
    # NOTE: Each thread maintains its own set of configuration, i.e. `dace.Config` is
    #   a thread local variable. This means it is safe to set values that are different
    #   for each thread.

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


@contextlib.contextmanager
def dace_context(**kwargs: Any) -> Generator[None, None, None]:
    """Create a DaCe configuration context and calls `set_dace_config()`.

    For more information see the description of `set_dace_config()`.
    """
    with dace.config.temporary_config():
        set_dace_config(**kwargs)
        yield
