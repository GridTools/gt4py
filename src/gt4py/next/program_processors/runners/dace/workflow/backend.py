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
class DaCeBackend(backend.Backend[Any]):
    """DaCe backend with support for injecting an external workspace at load time."""

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None

    def load_artifact(self, artifact: stages.CompilationArtifact) -> stages.ExecutableProgram:
        program = super().load_artifact(artifact)
        assert isinstance(program, gtx_wfddecoration.DaCeDecoratedProgram)
        # Inject the backend-level workspace so it is used when arguments are constructed.
        program.set_external_workspace(self.external_workspace or {})
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
    unstructured_horizontal_has_unit_stride: bool = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE,
    use_metrics: bool = True,
    use_zero_origin: bool = False,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
    use_stree_lowering: bool = False,
    apply_common_transforms: bool = False,
    name_postfix: str | None = None,
) -> backend.Backend:
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
        unstructured_horizontal_has_unit_stride: When the memory layout has unit stride
            in the horizontal dimension, replace the field stride symbol with '1'.
        use_metrics: Add SDFG instrumentation to collect the metric for stencil
            compute time.
        use_zero_origin: Can be set to `True` when all fields passed as program
            arguments have zero-based origin. This setting will skip generation
            of range start-symbols `_range_0` since they can be assumed to be zero.
        use_stree_lowering: Lower the GTIR program to SDFG through the schedule-tree
            intermediate representation; otherwise use the direct lowering.
        apply_common_transforms: Run the ITIR common-transforms pipeline (inlining,
            constant folding, CSE, ...) before lowering. Only supported with
            `use_stree_lowering=True`; otherwise the fieldview transforms are applied.
        name_postfix: Optional postfix for the backend name, appended to the name
            derived from the other configuration parameters.

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

    factory_kwargs: dict[str, Any] = dict(
        gpu=gpu,
        auto_optimize=auto_optimize,
        external_workspace=external_workspace,
        otf_workflow__bare_translation__async_sdfg_call=(async_sdfg_call if gpu else False),
        otf_workflow__bare_translation__auto_optimize_args=optimization_args,
        otf_workflow__bare_translation__unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
        otf_workflow__bare_translation__use_metrics=use_metrics,
        otf_workflow__bare_translation__disable_field_origin_on_program_arguments=use_zero_origin,
        otf_workflow__bare_translation__use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
        otf_workflow__bare_translation__use_stree_lowering=use_stree_lowering,
        otf_workflow__bare_translation__apply_common_transforms=apply_common_transforms,
    )
    if name_postfix is not None:
        factory_kwargs["name_postfix"] = name_postfix

    return DaCeBackendFactory(  # type: ignore[return-value] # factory-boy typing not precise enough
        **factory_kwargs,
    )


# Legacy backends: direct GTIR-to-SDFG lowering, without the ITIR
#  common-transforms pipeline.
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

# Schedule-tree lowering, without the ITIR common-transforms pipeline.
run_dace_stree_fview_cpu = make_dace_backend(
    gpu=False,
    auto_optimize=True,
    async_sdfg_call=False,
    use_stree_lowering=True,
    name_postfix="_stree_fview",
)
run_dace_stree_fview_cpu_noopt = make_dace_backend(
    gpu=False,
    auto_optimize=False,
    async_sdfg_call=False,
    use_stree_lowering=True,
    name_postfix="_stree_fview_noopt",
)
run_dace_stree_fview_gpu = make_dace_backend(
    gpu=True,
    auto_optimize=True,
    async_sdfg_call=True,
    use_stree_lowering=True,
    name_postfix="_stree_fview",
)

# Schedule-tree lowering, with the ITIR common-transforms pipeline (which also
#  performs operator fusion).
run_dace_stree_iview_cpu = make_dace_backend(
    gpu=False,
    auto_optimize=True,
    async_sdfg_call=False,
    use_stree_lowering=True,
    apply_common_transforms=True,
    name_postfix="_stree_iview",
)
run_dace_stree_iview_cpu_noopt = make_dace_backend(
    gpu=False,
    auto_optimize=False,
    async_sdfg_call=False,
    use_stree_lowering=True,
    apply_common_transforms=True,
    name_postfix="_stree_iview_noopt",
)
run_dace_stree_iview_gpu = make_dace_backend(
    gpu=True,
    auto_optimize=True,
    async_sdfg_call=True,
    use_stree_lowering=True,
    apply_common_transforms=True,
    name_postfix="_stree_iview",
)
