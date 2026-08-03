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
from typing import Any, Final

import factory

import gt4py.next.custom_layout_allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend, common, config
from gt4py.next.otf import stages
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
    factory as gtx_wfdfactory,
)


@dataclasses.dataclass(frozen=True)
class DaCeBackend(backend.Backend):
    """DaCe backend with support for injecting external resources at load time.

    These resources are intentionally kept out of the executor workflow: the workflow
    (and its `DaCeCompiler`) is pickled when compilation is offloaded to a worker
    process, while a `cupy.cuda.Stream` cannot be pickled. These resource are instead
    stored on the backend object, which lives in the calling process, and passed
    through to `DaCeCompilationArtifact.load()`.
    """

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None
    external_sync_stream: Any | None = None

    def load_artifact(self, artifact: stages.CompilationArtifact) -> stages.ExecutableProgram:
        program = super().load_artifact(artifact)
        assert isinstance(program, gtx_wfddecoration.DaCeDecoratedProgram)
        # Inject the backend-level workspace so it is used when arguments are constructed.
        program.set_external_workspace(self.external_workspace or {})
        program.set_external_sync_stream(self.external_sync_stream)
        return program


class DaCeBackendFactory(factory.Factory):
    """
    Workflow factory for the GTIR-DaCe backend.

    Several parameters are inherithed from `backend.Backend`, see below the specific ones.

    Args:
        auto_optimize: Enables the SDFG transformation pipeline.
    """

    class Meta:
        model = DaCeBackend

    class Params:
        name_device = "cpu"
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        device_type = core_defs.DeviceType.CPU
        otf_workflow = factory.SubFactory(
            gtx_wfdfactory.DaCeWorkflowFactory,
            cached_translation=True,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
        )
        auto_optimize = factory.Trait(name_postfix="_opt")

    name = factory.LazyAttribute(lambda o: f"run_dace_{o.name_device}{o.name_postfix}")
    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS
    external_workspace = None


def make_dace_backend(
    gpu: bool,
    auto_optimize: bool = True,
    async_sdfg_call: bool = True,
    optimization_args: dict[str, Any] | None = None,
    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None,
    external_sync_stream: Any | None = None,
    unstructured_horizontal_has_unit_stride: bool = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> DaCeBackend:
    """Customize the dace backend with the given configuration parameters.

    Args:
        gpu: Enable GPU transformations and code generation.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline, see `gt_auto_optimize()`.
        external_workspace: Workspace memory externally allocated, which is used
            for SDFG's transient arrays when `transient_memory_mode` is `EXTERNAL`.
        external_sync_stream: Optional `cupy.cuda.Stream` used as cross-stream
            synchronization anchor for `transient_memory_mode="external"` when
            `GT4PY_MAX_CONCURRENT_GPU_STREAMS > 0`. Stored on the backend object
            (not the picklable executor) and passed to the compiled program at
            load time.
        unstructured_horizontal_has_unit_stride: When the memory layout has unit stride
            in the horizontal dimension, replace the field stride symbol with '1'.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.
        use_zero_origin: Can be set to `True` when all fields passed as program
            arguments have zero-based origin. This setting will skip generation
            of range start-symbols `_range_0` since they can be assumed to be zero.

    Note that `gt_auto_optimize()` parameters that are derived from GT4Py configuration
    cannot be overriden, and therefore cannot appear here. Thus, this function will
    throw an exception if called with any argument included in `gt_optimization_args`.

    Returns:
        A dace backend with custom configuration for the target device.
    """

    # The `gt_optimization_args` set contains the parameters of `gt_auto_optimize()`
    # that are derived from the gt4py configuration, and therefore cannot be customized.
    gt_optimization_args: Final[set[str]] = {"gpu", "constant_symbols", "unit_strides_kind"}

    if optimization_args is None:
        optimization_args = {}
    elif optimization_args and not auto_optimize:
        warnings.warn("Optimizations args given, but auto-optimize is disabled.", stacklevel=2)
    elif intersect_args := gt_optimization_args.intersection(optimization_args.keys()):
        raise ValueError(
            f"The following optimization arguments cannot be overriden: {intersect_args}."
        )

    # Set `unit_strides_kind` based on the gt4py env configuration.
    optimization_args = optimization_args | {
        "unit_strides_kind": common.DimensionKind.HORIZONTAL
        if unstructured_horizontal_has_unit_stride
        else None
    }

    if external_workspace is None:
        if (
            optimization_args.get("transient_memory_mode")
            is gtx_transformations.TransientMemoryMode.EXTERNAL
        ):
            raise ValueError(
                "External memory workspace must be provided when 'transient_memory_mode' is 'EXTERNAL'."
            )
    elif transient_memory_mode := optimization_args.get("transient_memory_mode"):
        if transient_memory_mode is not gtx_transformations.TransientMemoryMode.EXTERNAL:
            warnings.warn(
                f"External memory workspace provided but 'transient_memory_mode' is '{transient_memory_mode}', it requires '{gtx_transformations.TransientMemoryMode.EXTERNAL}'.",
                stacklevel=2,
            )
    else:
        optimization_args["transient_memory_mode"] = (
            gtx_transformations.TransientMemoryMode.EXTERNAL
        )

    if (
        optimization_args.get("transient_memory_mode")
        is gtx_transformations.TransientMemoryMode.POOL
        and config.MAX_CONCURRENT_GPU_STREAMS > 0
    ):
        raise ValueError(
            "DaCe backend does not implement in-order memory allocations and "
            "multi-stream scheduling together. Use `transient_memory_mode='external'` "
            "or 'persistent', or set `GT4PY_MAX_CONCURRENT_GPU_STREAMS=0`."
        )

    base_backend = DaCeBackendFactory(
        gpu=gpu,
        auto_optimize=auto_optimize,
        external_workspace=external_workspace,
        otf_workflow__bare_translation__async_sdfg_call=(async_sdfg_call if gpu else False),
        otf_workflow__bare_translation__auto_optimize_args=optimization_args,
        otf_workflow__bare_translation__unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        otf_workflow__bare_translation__use_metrics=use_metrics,
        otf_workflow__bare_translation__disable_field_origin_on_program_arguments=use_zero_origin,
        otf_workflow__bare_translation__use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )
    return DaCeBackend(
        name=base_backend.name,  # type: ignore[arg-type] # factory-boy typing not precise enough
        executor=base_backend.executor,  # type: ignore[arg-type]
        allocator=base_backend.allocator,
        transforms=base_backend.transforms,
        external_sync_stream=external_sync_stream,
    )


run_dace_cpu = make_dace_backend(
    gpu=False,
    auto_optimize=True,
    async_sdfg_call=False,
)
run_dace_cpu_noopt = make_dace_backend(
    gpu=False,
    auto_optimize=False,
    async_sdfg_call=False,
)

run_dace_gpu = make_dace_backend(
    gpu=True,
    auto_optimize=True,
    async_sdfg_call=True,
)
run_dace_gpu_noopt = make_dace_backend(
    gpu=True,
    auto_optimize=False,
    async_sdfg_call=True,
)
