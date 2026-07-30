# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

from gt4py.next import backend, config
from gt4py.next.otf import artifacts, stages, workflow
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
    factory as gtx_wfdfactory,
)


@dataclasses.dataclass(frozen=True)
class DaCeLoadingStep:
    """
    Loading step that injects an external workspace into the loaded program.

    The workspace is owned by the caller, not by the toolchain; it is installed
    onto the program wrapper before its first call, so that it is used when the
    SDFG argument vector is constructed.
    """

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None

    def __call__(self, artifact: artifacts.CompilationArtifact) -> artifacts.ExecutableProgram:
        program = artifacts.load_artifact(artifact)
        assert isinstance(program, gtx_wfddecoration.DaCeDecoratedProgram)
        program.set_external_workspace(self.external_workspace or {})
        return program


def make_dace_toolchain(
    cfg: gtx_wfdfactory.DaCeConfig | None = None,
    /,
    *,
    name_postfix: str = "",
    translation: Callable[
        [gtx_wfdfactory.DaCeConfig], stages.TranslationStep
    ] = gtx_wfdfactory.make_dace_translator,
    bindings: Callable[
        [gtx_wfdfactory.DaCeConfig],
        workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource],
    ] = gtx_wfdfactory.make_dace_bindings,
    compilation: Callable[
        [gtx_wfdfactory.DaCeConfig],
        workflow.Step[artifacts.ExtensionSource, artifacts.CompilationArtifact],
    ] = gtx_wfdfactory.make_dace_compiler,
) -> backend.Toolchain:
    """
    Build a DaCe toolchain.

    Args:
        cfg: The toolchain configuration. Defaults to `DaCeConfig()`.
        name_postfix: Appended to the toolchain name, which must stay unique.
        translation: Builder of the translation step, see
            `make_dace_compile_workflow`.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The configured toolchain.

    Raises:
        ValueError: If a step builder returns a step configured for a device
            other than `cfg.device_type`.
    """
    if cfg is None:
        cfg = gtx_wfdfactory.DaCeConfig()

    return backend.Toolchain(
        name=f"run_dace_{cfg.device_name}{'_opt' if cfg.auto_optimize else ''}{name_postfix}",
        backend=gtx_wfdfactory.make_dace_compile_workflow(
            cfg, translation=translation, bindings=bindings, compilation=compilation
        ),
        allocator=cfg.make_allocator(),
        frontend=backend.DEFAULT_TRANSFORMS,
        loading=DaCeLoadingStep(cfg.external_workspace),
    )


def make_dace_backend(
    gpu: bool,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, Any] | None = None,
    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None,
    unstructured_horizontal_has_unit_stride: bool = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> backend.Toolchain:
    """Customize the dace backend with the given configuration parameters.

    A flat-keyword front end for `make_dace_toolchain`, kept for existing
    callers: it builds the `DaCeConfig` and the translation step builder from
    its arguments.

    Args:
        gpu: Enable GPU transformations and code generation.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline, see `gt_auto_optimize()`.
        external_workspace: Workspace memory externally allocated, which is used
            for SDFG's transient arrays when `transient_memory_mode` is `EXTERNAL`.
        unstructured_horizontal_has_unit_stride: When the memory layout has unit stride
            in the horizontal dimension, replace the field stride symbol with '1'.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.
        use_zero_origin: Can be set to `True` when all fields passed as program
            arguments have zero-based origin. This setting will skip generation
            of range start-symbols `_range_0` since they can be assumed to be zero.
        use_max_domain_range_on_unstructured_shift: See `DaCeTranslator`.

    Note that `gt_auto_optimize()` parameters that are derived from GT4Py configuration
    cannot be overriden, and therefore cannot appear in `optimization_args`.

    Returns:
        A dace backend with custom configuration for the target device.

    Raises:
        ValueError: If `optimization_args` sets a parameter derived from the
            configuration, or requests the `EXTERNAL` transient memory mode
            without an `external_workspace`.
    """
    return make_dace_toolchain(
        gtx_wfdfactory.DaCeConfig(
            gpu=gpu,
            auto_optimize=auto_optimize,
            external_workspace=external_workspace,
            unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        ),
        translation=functools.partial(
            gtx_wfdfactory.make_dace_translator,
            optimization_args=optimization_args,
            async_sdfg_call=async_sdfg_call,
            use_metrics=use_metrics,
            use_zero_origin=use_zero_origin,
            use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
        ),
    )


run_dace_cpu = make_dace_toolchain(gtx_wfdfactory.DaCeConfig(auto_optimize=True))
run_dace_cpu_noopt = make_dace_toolchain(gtx_wfdfactory.DaCeConfig(auto_optimize=False))

run_dace_gpu = make_dace_toolchain(gtx_wfdfactory.DaCeConfig(gpu=True, auto_optimize=True))
run_dace_gpu_noopt = make_dace_toolchain(gtx_wfdfactory.DaCeConfig(gpu=True, auto_optimize=False))
