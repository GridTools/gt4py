# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

from gt4py.next import backend, config
from gt4py.next.otf import artifacts
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
    factory as gtx_wfdfactory,
)


@dataclasses.dataclass(frozen=True)
class DaCeBackend(backend.Backend[Any]):
    """DaCe backend with support for injecting an external workspace at load time."""

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None

    def load_artifact(self, artifact: artifacts.CompilationArtifact) -> artifacts.ExecutableProgram:
        program = super().load_artifact(artifact)
        assert isinstance(program, gtx_wfddecoration.DaCeDecoratedProgram)
        # Inject the backend-level workspace so it is used when arguments are constructed.
        program.set_external_workspace(self.external_workspace or {})
        return program


def make_dace_backend(
    gpu: bool,
    auto_optimize: bool = True,
    *,
    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None,
    unstructured_horizontal_has_unit_stride: bool | None = None,
    cached_translation: bool = True,
    cmake_build_type: config.CMakeBuildType | None = None,
    translation: gtx_wfdfactory.DaCeTranslationOptions | None = None,
    compilation: gtx_wfdfactory.DaCeCompilationOptions | None = None,
) -> DaCeBackend:
    """Customize the dace backend with the given configuration parameters.

    Settings shared by several steps are keyword arguments; the step-local
    settings of the translation and compilation steps are passed as dicts, see
    `make_dace_compile_workflow`.

    Args:
        gpu: Enable GPU transformations and code generation.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        external_workspace: Workspace memory externally allocated, which is used
            for SDFG's transient arrays when `transient_memory_mode` is `EXTERNAL`.
        unstructured_horizontal_has_unit_stride: When the memory layout has unit stride
            in the horizontal dimension, replace the field stride symbol with '1'.
            Defaults to the value in `config`.
        cached_translation: Wrap the translation step in a persistent cache.
        cmake_build_type: Build type of the generated project. Defaults to the
            value in `config`.
        translation: Step-local settings of the translation step, see
            `DaCeTranslator`. When an `external_workspace` is given and
            `auto_optimize_args` sets no `transient_memory_mode`, it defaults to
            `EXTERNAL`.
        compilation: Step-local settings of the compilation step.

    Returns:
        A dace backend with custom configuration for the target device.

    Raises:
        ValueError: If `auto_optimize_args` requests the `EXTERNAL` transient
            memory mode without an `external_workspace`, or sets a parameter the
            translation step derives itself.
    """
    if unstructured_horizontal_has_unit_stride is None:
        unstructured_horizontal_has_unit_stride = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
    if translation is None:
        translation = gtx_wfdfactory.DaCeTranslationOptions()

    # The external workspace belongs to the backend, which injects it at load
    # time, so the backend owns its coupling to the transient memory mode.
    optimization_args = dict(translation.get("auto_optimize_args") or {})
    transient_memory_mode = optimization_args.get("transient_memory_mode")
    if external_workspace is None:
        if transient_memory_mode is gtx_transformations.TransientMemoryMode.EXTERNAL:
            raise ValueError(
                "External memory workspace must be provided when 'transient_memory_mode' is 'EXTERNAL'."
            )
    elif transient_memory_mode is None:
        if auto_optimize:
            optimization_args["transient_memory_mode"] = (
                gtx_transformations.TransientMemoryMode.EXTERNAL
            )
            translation = translation | gtx_wfdfactory.DaCeTranslationOptions(
                auto_optimize_args=optimization_args
            )
    elif transient_memory_mode is not gtx_transformations.TransientMemoryMode.EXTERNAL:
        warnings.warn(
            f"External memory workspace provided but 'transient_memory_mode' is '{transient_memory_mode}', it requires '{gtx_transformations.TransientMemoryMode.EXTERNAL}'.",
            stacklevel=2,
        )

    device_type, allocator = backend.select_device(gpu)

    return DaCeBackend(
        name=f"run_dace_{'gpu' if gpu else 'cpu'}{'_opt' if auto_optimize else ''}",
        executor=gtx_wfdfactory.make_dace_compile_workflow(
            device_type=device_type,
            auto_optimize=auto_optimize,
            cached_translation=cached_translation,
            cmake_build_type=cmake_build_type,
            unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
            translation=translation,
            compilation=compilation,
        ),
        allocator=allocator,
        transforms=backend.DEFAULT_TRANSFORMS,
        external_workspace=external_workspace,
    )


run_dace_cpu = make_dace_backend(gpu=False, auto_optimize=True)
run_dace_cpu_noopt = make_dace_backend(gpu=False, auto_optimize=False)

run_dace_gpu = make_dace_backend(gpu=True, auto_optimize=True)
run_dace_gpu_noopt = make_dace_backend(gpu=True, auto_optimize=False)
