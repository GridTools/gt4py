# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import Callable, Final

import factory

import gt4py.next.allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.next import backend, common, config, factory_utils
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors.runners.dace.workflow.factory import DaCeWorkflowFactory


class DaCeBackendFactory(factory_utils.Factory):
    """
    Configurable factory for DaCe backend.

    Parameters:
        gpu: Enable GPU transformations and code generation.
        cached: Cache the lowered SDFG as a JSON file and the compiled programs.
        auto_optimize: Enable the SDFG auto-optimize pipeline.
        async_sdfg_call: Make an asynchronous SDFG call on GPU to allow overlapping
            of GPU kernel execution with the Python driver code.
        optimization_args: A `dict` containing configuration parameters for
            the SDFG auto-optimize pipeline, see `gt_auto_optimize()`.
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

    class Meta:
        model = backend.Backend

    class Params:
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
        )
        cached: bool = True
        auto_optimize: bool = True
        async_sdfg_call: bool = factory.SelfAttribute(".gpu")  # type: ignore[assignment]
        use_metrics: bool = True
        use_zero_origin: bool = False
        device_type = core_defs.DeviceType.CPU
        hash_function = stages.compilation_hash

        @factory_utils.dynamic_transformer(default=factory.Dict({}))
        def optimization_args(self, optimization_args: dict) -> dict:
            # The `gt_optimization_args` set contains the parameters of `gt_auto_optimize()`
            # that are derived from the gt4py configuration, and therefore cannot be customized.
            gt_optimization_args: Final[set[str]] = {"gpu", "constant_symbols", "unit_strides_kind"}

            if optimization_args and not self.auto_optimize:
                warnings.warn(
                    "Optimizations args given, but auto-optimize is disabled.", stacklevel=2
                )
            elif intersect_args := gt_optimization_args.intersection(optimization_args.keys()):
                raise ValueError(
                    f"The following optimization arguments cannot be overriden: {intersect_args}."
                )

            # Set `unit_strides_kind` based on the gt4py env configuration.
            return optimization_args | {
                "unit_strides_kind": common.DimensionKind.HORIZONTAL
                if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                else None
            }

    @factory.lazy_attribute
    def name(self: factory.builder.Resolver) -> str:
        name = "run_dace_"
        name += "gpu" if self.gpu else "cpu"
        if self.auto_optimize:
            name += "_opt"
        if self.cached:
            name += "_cached"
        return name

    @factory_utils.dynamic_transformer(
        default=factory.SubFactory(
            DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
            translation__async_sdfg_call=factory.SelfAttribute("...async_sdfg_call"),
            translation__auto_optimize_args=factory.SelfAttribute("...optimization_args"),
            translation__use_metrics=factory.SelfAttribute("...use_metrics"),
            translation__disable_field_origin_on_program_arguments=factory.SelfAttribute(
                "...use_zero_origin"
            ),
        )
    )
    def executor(self: factory.builder.Resolver, value: Callable) -> Callable:
        if self.cached:
            return workflow.CachedStep(value, hash_function=self.hash_function)
        return value

    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


make_dace_backend = DaCeBackendFactory


run_dace_cpu = make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=True,
)
run_dace_cpu_noopt = make_dace_backend(
    gpu=False,
    cached=False,
    auto_optimize=False,
)
run_dace_cpu_cached = make_dace_backend(
    gpu=False,
    cached=True,
    auto_optimize=True,
)

run_dace_gpu = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=True,
)
run_dace_gpu_noopt = make_dace_backend(
    gpu=True,
    cached=False,
    auto_optimize=False,
)
run_dace_gpu_cached = make_dace_backend(
    gpu=True,
    cached=True,
    auto_optimize=True,
)
