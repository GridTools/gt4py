# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import dace
import factory

from gt4py._core import definitions as core_defs
from gt4py.next import common, config, metrics
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import languages, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace import (
    gtir_to_sdfg,
    transformations as gtx_transformations,
    utils as gtx_dace_utils,
)
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon
from gt4py.next.type_system import type_specifications as ts


def find_constant_symbols(
    ir: itir.Program,
    sdfg: dace.SDFG,
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, int]:
    """Helper function to find symbols to replace with constant values."""
    constant_symbols: dict[str, int] = {}
    if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE:
        # Search the stride symbols corresponding to the horizontal dimension
        for p in ir.params:
            if isinstance(p.type, ts.FieldType):
                h_dims = [dim for dim in p.type.dims if dim.kind == common.DimensionKind.HORIZONTAL]
                if len(h_dims) == 0:
                    continue
                elif len(h_dims) == 1:
                    dim = h_dims[0]
                else:
                    raise NotImplementedError(
                        f"Unsupported field with multiple horizontal dimensions '{p}'."
                    )
                if isinstance(p.type.dtype, ts.ListType):
                    assert p.type.dtype.offset_type is not None
                    full_dims = common.order_dimensions([*p.type.dims, p.type.dtype.offset_type])
                    dim_index = full_dims.index(dim)
                else:
                    dim_index = p.type.dims.index(dim)
                stride_name = gtx_dace_utils.field_stride_symbol_name(p.id, dim_index)
                constant_symbols[stride_name] = 1
        # Same for connectivity tables, for which the first dimension is always horizontal
        for conn, desc in sdfg.arrays.items():
            if gtx_dace_utils.is_connectivity_identifier(conn, offset_provider_type):
                assert not desc.transient
                stride_name = gtx_dace_utils.field_stride_symbol_name(conn, 0)
                constant_symbols[stride_name] = 1

    return constant_symbols


def make_sdfg_call_async(sdfg: dace.SDFG, gpu: bool) -> None:
    """Configure an SDFG to immediately return once all work has been scheduled.

    This means that `CompiledSDFG.fast_call()` will return immediately after all
    computations have been _scheduled_ on the device. This function only has an effect
    for work that runs on the GPU. Furthermore, all work is scheduled on the
    default stream.

    Todo: Revisit this function once DaCe changes its behaviour in this regard.
    """

    # This is only a problem on GPU.
    # TODO(phimuell): Figuring out what about OpenMP.
    if not gpu:
        return

    assert dace.Config.get("compiler.cuda.max_concurrent_streams") == -1, (
        f"Expected `max_concurrent_streams == -1` but it was `{dace.Config.get('compiler.cuda.max_concurrent_streams')}`."
    )

    # NOTE: We are using the default stream this means that _**currently**_ the launch is
    #   already asynchronous, see [DaCe issue#2120](https://github.com/spcl/dace/issues/2120)
    #   for more. However, DaCe still [generates streams internally](https://github.com/spcl/dace/blob/54c935cfe74a52c5107dc91680e6201ddbf86821/dace/codegen/targets/cuda.py#L467).
    #   Thus to be absolutely sure we will no set all streams, DaCe uses to the default
    #   stream.
    # NOTE: Another important note here is, that it looks as if no synchronization
    #   what soever between states is generated if the default stream is used. This
    #   might lead to problems if a GPU kernel computes something that is needed for a
    #   condition of an interstate edge. However, we should not have that case.
    # TODO(phimuell, edopao): Make sure if this is really the case.
    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."
    sdfg.append_init_code(
        f"__dace_gpu_set_all_streams(__state, {dace_gpu_backend}StreamDefault);",
        location="cuda",
    )


def _has_gpu_schedule(sdfg: dace.SDFG) -> bool:
    """Check if any node (e.g. maps) of the given SDFG is scheduled on GPU."""
    return any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )


def add_instrumentation(sdfg: dace.SDFG, gpu: bool) -> None:
    """
    Instrument SDFG with measurement of total execution time.

    We measure the execution time of one GT4Py program by instrumenting the top-level
    SDFG with a cpp timer (std::chrono). This timer measures only the computation
    time, it does not include the overhead of calling the SDFG from Python.

    The execution time is measured in seconds and represented as a 'float64' value.
    It is returned from the SDFG as a one-element array in the '__return' data node.
    """
    output, _ = sdfg.add_array("__return", [1], dace.float64)
    start_time, _ = sdfg.add_scalar("gt_start_time", dace.float64, transient=True)
    metrics_level = sdfg.add_symbol(gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL, dace.int32)

    #### 1. Synchronize the CUDA device, in order to wait for kernels completion.
    # Even when the target device is GPU, it can happen that dace emits code without
    # GPU kernels. In this case, the cuda headers are not imported and the SDFG is
    # compiled as plain C++. Therefore, we also check here the schedule of SDFG maps.
    if gpu and _has_gpu_schedule(sdfg):
        dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
        assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."

        # NOTE: We should actually wrap the `DeviceSynchronize` function inside a
        #   `DACE_GPU_CHECK()` macro. However, this only works in GPU context, but
        #   here we are in CPU context. Thus we cannot do it.
        sync_code = f"{dace_gpu_backend}DeviceSynchronize();"
        has_side_effects = True

    else:
        sync_code = ""
        has_side_effects = False

    #### 2. Timestamp the SDFG entry point.
    entry_state = sdfg.add_state("gt_timer_entry")
    begin_state = sdfg.add_state_after(
        entry_state, "gt_timer_begin", condition=f"{metrics_level} >= {metrics.PERFORMANCE}"
    )

    for source_state in sdfg.source_nodes():
        if source_state is entry_state:
            continue
        sdfg.add_edge(
            entry_state,
            source_state,
            dace.InterstateEdge(condition=f"{metrics_level} < {metrics.PERFORMANCE}"),
        )
        sdfg.add_edge(begin_state, source_state, dace.InterstateEdge())
    assert sdfg.out_degree(begin_state) > 0
    assert sdfg.out_degree(entry_state) > 1

    tlet_start_timer = begin_state.add_tasklet(
        "gt_start_timer",
        inputs={},
        outputs={"time"},
        code="""\
time = static_cast<double>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
) / 1e9;
        """,
        language=dace.dtypes.Language.CPP,
    )
    begin_state.add_edge(
        tlet_start_timer,
        "time",
        begin_state.add_access(start_time),
        None,
        dace.Memlet(f"{start_time}[0]"),
    )

    #### 3. Collect the SDFG end timestamp and produce the compute metric.
    end_state = sdfg.add_state("gt_timer_end")
    for sink_state in sdfg.sink_nodes():
        if sink_state is end_state:
            continue
        sdfg.add_edge(
            sink_state,
            end_state,
            dace.InterstateEdge(condition=f"{metrics_level} >= {metrics.PERFORMANCE}"),
        )
    assert sdfg.in_degree(end_state) > 0

    tlet_stop_timer = end_state.add_tasklet(
        "gt_stop_timer",
        inputs={"run_cpp_start_time"},
        outputs={"duration"},
        code=sync_code
        + """
double run_cpp_end_time = static_cast<double>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
) / 1e9;
duration = run_cpp_end_time - run_cpp_start_time;
        """,
        language=dace.dtypes.Language.CPP,
        side_effects=has_side_effects,
    )
    end_state.add_edge(
        end_state.add_access(start_time),
        None,
        tlet_stop_timer,
        "run_cpp_start_time",
        dace.Memlet(f"{start_time}[0]"),
    )
    end_state.add_edge(
        tlet_stop_timer, "duration", end_state.add_access(output), None, dace.Memlet(f"{output}[0]")
    )


def make_sdfg_call_sync(sdfg: dace.SDFG, gpu: bool) -> None:
    """Process the SDFG such that the call is synchronous.

    This means that `CompiledSDFG.fast_call()` will return only after all computations
    have _finished_ and the results are available. This function only has an effect for
    work that runs on the GPU. Furthermore, all work is scheduled on the default stream.

    Todo: Revisit this function once DaCe changes its behaviour in this regard.
    """

    if not gpu:
        # This is only a problem on GPU. Dace uses OpenMP on CPU and
        # the OpenMP parallel region creates a synchronization point.
        return
    elif not _has_gpu_schedule(sdfg):
        # Even when the target device is GPU, it can happen that dace
        # emits code without GPU kernels. In this case, the cuda headers
        # are not imported and the SDFG is compiled as plain C++.
        return

    assert dace.Config.get("compiler.cuda.max_concurrent_streams") == -1, (
        f"Expected `max_concurrent_streams == -1` but it was `{dace.Config.get('compiler.cuda.max_concurrent_streams')}`."
    )

    # If we are using the default stream, things are a bit simpler/harder. For some
    #  reasons when using the default stream, DaCe seems to skip _all_ synchronization,
    #  for more see [DaCe issue#2120](https://github.com/spcl/dace/issues/2120).
    #  Thus the `CompiledSDFG.fast_call()` call is truly asynchronous, i.e. just
    #  launches the kernels and then exist. Thus we have to add a synchronization
    #  at the end to have a synchronous call. We can not use `SDFG.append_exit_code()`
    #  because that code is only run at the `exit()` stage, not after a call. Thus we
    #  will generate an SDFGState that contains a Tasklet with the sync call.
    sync_state = sdfg.add_state("sync_state")
    for sink_state in sdfg.sink_nodes():
        if sink_state is sync_state:
            continue
        sdfg.add_edge(sink_state, sync_state, dace.InterstateEdge())
    assert sdfg.in_degree(sync_state) > 0

    # NOTE: Since the synchronization is done through the Tasklet explicitly,
    #   we can disable synchronization for the last state. Might be useless.
    sync_state.nosync = True

    # NOTE: We should actually wrap the `StreamSynchronize` function inside a
    #   `DACE_GPU_CHECK()` macro. However, this only works in GPU context, but
    #   here we are in CPU context. Thus we can not do it.
    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."
    sync_state.add_tasklet(
        "sync_tlet",
        inputs=set(),
        outputs=set(),
        code=f"{dace_gpu_backend}StreamSynchronize({dace_gpu_backend}StreamDefault);",
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )

    # DaCe [still generates a stream](https://github.com/spcl/dace/blob/54c935cfe74a52c5107dc91680e6201ddbf86821/dace/codegen/targets/cuda.py#L467)
    #  despite not using it. Thus to be absolutely sure, we will not set that stream
    #  to the default stream.
    sdfg.append_init_code(
        f"__dace_gpu_set_all_streams(__state, {dace_gpu_backend}StreamDefault);",
        location="cuda",
    )


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    device_type: core_defs.DeviceType
    auto_optimize: bool
    async_sdfg_call: bool
    use_metrics: bool
    disable_itir_transforms: bool = False
    disable_field_origin_on_program_arguments: bool = False

    # auto-optimize arguments
    gpu_block_size: tuple[int, int, int] = (32, 8, 1)
    make_persistent: bool = False
    use_memory_pool: bool = False
    blocking_dim: Optional[common.Dimension] = None
    blocking_size: int = 10
    validate: bool = False
    validate_all: bool = False

    def generate_sdfg(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dace.SDFG:
        with gtx_wfdcommon.dace_context(device_type=self.device_type):
            return self._generate_sdfg_without_configuring_dace(*args, **kwargs)

    def _generate_sdfg_without_configuring_dace(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        if not self.disable_itir_transforms:
            ir = itir_transforms.apply_fieldview_transforms(ir, offset_provider=offset_provider)
        offset_provider_type = common.offset_provider_to_type(offset_provider)
        on_gpu = self.device_type != core_defs.DeviceType.CPU

        if self.use_memory_pool and not on_gpu:
            raise NotImplementedError("Memory pool only available for GPU device.")

        sdfg = gtir_to_sdfg.build_sdfg_from_gtir(
            ir,
            offset_provider_type,
            column_axis,
            disable_field_origin_on_program_arguments=self.disable_field_origin_on_program_arguments,
        )

        if self.auto_optimize:
            unit_strides_kind = (
                common.DimensionKind.HORIZONTAL
                if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                else None  # let `gt_auto_optimize` select `unit_strides_kind` based on `gpu` argument
            )
            constant_symbols = find_constant_symbols(ir, sdfg, offset_provider_type)
            gtx_transformations.gt_auto_optimize(
                sdfg,
                gpu=on_gpu,
                gpu_block_size=self.gpu_block_size,
                unit_strides_kind=unit_strides_kind,
                constant_symbols=constant_symbols,
                assume_pointwise=True,
                make_persistent=self.make_persistent,
                gpu_memory_pool=self.use_memory_pool,
                blocking_dim=self.blocking_dim,
                blocking_size=self.blocking_size,
                validate=self.validate,
                validate_all=self.validate_all,
            )
        elif on_gpu:
            # We run simplify to bring the SDFG into a canonical form that the GPU transformations
            # can handle. This is a workaround for an issue with scalar expressions that are
            # promoted to symbolic expressions and computed on the host (CPU), but the intermediate
            # result is written to a GPU global variable (https://github.com/spcl/dace/issues/1773).
            gtx_transformations.gt_simplify(sdfg)
            gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        if self.async_sdfg_call:
            make_sdfg_call_async(sdfg, on_gpu)
        else:
            make_sdfg_call_sync(sdfg, on_gpu)

        if self.use_metrics:
            add_instrumentation(sdfg, on_gpu)

        return sdfg

    def __call__(
        self, inp: stages.CompilableProgram
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the GTIR definition."""
        program: itir.Program = inp.data
        assert isinstance(program, itir.Program)

        sdfg = self.generate_sdfg(
            program,
            inp.args.offset_provider,  # TODO(havogt): should be offset_provider_type once the transformation don't require run-time info
            inp.args.column_axis,
        )

        arg_types = inp.args.args

        program_parameters = tuple(
            interface.Parameter(param.id, arg_type)
            for param, arg_type in zip(program.params, arg_types)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, program_parameters),
                # Set 'hash=True' to compute the SDFG hash and store it in the JSON.
                #   We compute the hash in order to refresh `cfg_list` on the SDFG,
                #   which makes the JSON serialization stable.
                source_code=sdfg.to_json(hash=True),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=languages.LanguageSettings(
                    formatter_key="", formatter_style="", file_extension="sdfg"
                ),
            )
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator
