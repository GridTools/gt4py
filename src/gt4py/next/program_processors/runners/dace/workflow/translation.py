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
from gt4py.next import common
from gt4py.next.instrumentation import metrics
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners.dace import (
    lowering as gtx_dace_lowering,
    sdfg_args as gtx_dace_args,
    transformations as gtx_transformations,
)
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon
from gt4py.next.type_system import type_specifications as ts


def find_constant_symbols(
    ir: itir.Program,
    sdfg: dace.SDFG,
    offset_provider_type: common.OffsetProviderType,
    disable_field_origin_on_program_arguments: bool,
    unstructured_horizontal_has_unit_stride: bool,
) -> dict[str, int]:
    """Helper function to find symbols to replace with constant values."""
    constant_symbols: dict[str, int] = {}

    if unstructured_horizontal_has_unit_stride:
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
                sdfg_stride_symbol = gtx_dace_args.field_stride_symbol(str(p.id), dim)
                constant_symbols[sdfg_stride_symbol.name] = 1
        # Same for connectivity tables, for which the first dimension is always horizontal
        for offset, conn_type in offset_provider_type.items():
            if (
                isinstance(conn_type, common.NeighborConnectivityType)
                and (conn_id := gtx_dace_args.connectivity_identifier(offset)) in sdfg.arrays
            ):
                assert not sdfg.arrays[conn_id].transient
                assert conn_type.source_dim.kind == common.DimensionKind.HORIZONTAL
                sdfg_stride_symbol = gtx_dace_args.field_stride_symbol(
                    conn_id, conn_type.source_dim, offset_provider_type
                )
                constant_symbols[sdfg_stride_symbol.name] = 1

    if disable_field_origin_on_program_arguments:
        # collect symbols used as range start for all program arguments
        for p in ir.params:
            if isinstance(p.type, ts.TupleType):
                psymbols = [
                    sym
                    for sym in gtx_dace_lowering.flatten_tuple_fields(p.id, p.type)
                    if isinstance(sym.type, ts.FieldType)
                ]
            elif isinstance(p.type, ts.FieldType):
                psymbols = [p]
            else:
                psymbols = []
            for psymbol in psymbols:
                assert isinstance(psymbol.type, ts.FieldType)
                if len(psymbol.type.dims) == 0:
                    # zero-dimensional field
                    continue
                # set all range start symbols to constant value 0
                sdfg_origin_symbols = [
                    gtx_dace_args.range_start_symbol(str(psymbol.id), dim)
                    for dim in psymbol.type.dims
                ]
                constant_symbols |= {sdfg_symbol.name: 0 for sdfg_symbol in sdfg_origin_symbols}

    return constant_symbols


def _has_gpu_schedule(sdfg: dace.SDFG) -> bool:
    """Check if any node (e.g. maps) of the given SDFG is scheduled on GPU."""
    return any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )


def add_synchronization(sdfg: dace.SDFG, *, gpu: bool, blocking: bool, n_streams: int) -> None:
    """Synchronize SDFG execution with an externally managed stream.

    Args:
        sdfg: The DaCe SDFG to modify.
        gpu: ``True`` when the target device is a GPU; CPU targets are a no-op.
        blocking: Select whether the SDFG call should block before returning,
            until the SDFG execution has completed; or it should return immediately
            without blocking, thus letting the GPU kernels run on the GPU device
            while the control flow returns to the caller.
        n_streams: Number of concurrent internal GPU streams to use.
            ``0`` disables multi-stream scheduling and executes asynchronously on
            the default CUDA/HIP stream. Values ``>= 1`` enable DaCe's internal
            stream pool and add event-based synchronization with an external
            stream. The external stream pointer is passed at runtime as the SDFG
            argument ``gtx_wfdcommon.SDFG_ARG_EXTERNAL_SYNC_STREAM``; ``0`` means
            the default stream.

    This function operates on the top-level SDFG produced for a GT4Py program.
    It rewrites the control flow around ``sdfg.start_block`` and the original
    sink nodes, which is only valid for that top-level SDFG.

    This function inserts entry and exit tasklets that use CUDA/HIP events to
    establish a bidirectional barrier between the external sync stream and the
    internal DaCe streams when ``n_streams >= 1``:

    - The entry tasklet records events on the external stream and makes every
      internal stream wait on them, so that internal kernels do not start until
      the external stream has finished any prior work on the workspace.
    - The exit tasklet records per-stream done events on each internal stream
      and makes the external stream wait on them, so the external stream does
      not proceed until all internal streams have finished the current call.

    Cross-call ordering on the external stream is the caller's responsibility;
    GT4Py only orders the internal streams with respect to the external stream
    within each SDFG call.

    The tasklets reference the SDFG symbol `gtx_wfdcommon.SDFG_ARG_EXTERNAL_SYNC_STREAM`
    (the external stream pointer, or ``0`` for the default stream).

    This function is a no-op unless the target is GPU, the SDFG contains GPU
    schedules, and ``n_streams >= 1`` (or ``blocking`` is ``True``).
    """
    if not gpu:
        # CPU targets are always synchronous; no synchronization tasklets are needed.
        return

    if not _has_gpu_schedule(sdfg):
        # No GPU kernels means no CUDA headers are imported; skip sync tasklets.
        return

    dace_gpu_backend = dace.Config.get("compiler.cuda.backend")
    assert dace_gpu_backend in ["cuda", "hip"], f"GPU backend '{dace_gpu_backend}' is unknown."

    original_start = sdfg.start_block

    if n_streams == 0:
        if not blocking:
            return  # Fully asynchronous execution: no tasklets needed.

        stream_arg = f"{dace_gpu_backend}StreamDefault"
        entry_code = exit_code = "/* No synchronization needed, using default stream */"
    else:
        # Add a scalar symbol for the stream handle. It is passed as a 64-bit integer
        # pointer from Python and cast to the appropriate CUDA/HIP handle type inside
        # the generated tasklets.
        stream_arg = gtx_wfdcommon.SDFG_ARG_EXTERNAL_SYNC_STREAM
        sdfg.add_symbol(stream_arg, dace.uint64)

        # Asynchronous entry barrier: make the internal streams wait on the external stream.
        entry_code = "\n".join(
            [
                "for (int i = 0; i < __state->gpu_context->num_events; ++i) {",
                f"    {dace_gpu_backend}EventRecord(__state->gpu_context->events[i], "
                f"({dace_gpu_backend}Stream_t){stream_arg});",
                "}",
                "for (int i = 0; i < __state->gpu_context->num_events; ++i) {",
                f"    {dace_gpu_backend}StreamWaitEvent(__state->gpu_context->streams[i], "
                f"__state->gpu_context->events[i], 0);",
                "}",
            ]
        )

        # Asynchronous exit barrier: record per-stream done events and make the
        # external stream wait on them.
        exit_code = "\n".join(
            [
                "for (int i = 0; i < __state->gpu_context->num_events; ++i) {",
                f"    {dace_gpu_backend}EventRecord(__state->gpu_context->events[i], "
                f"__state->gpu_context->streams[i]);",
                "}",
                "for (int i = 0; i < __state->gpu_context->num_events; ++i) {",
                f"    {dace_gpu_backend}StreamWaitEvent(({dace_gpu_backend}Stream_t){stream_arg}, "
                f"__state->gpu_context->events[i], 0);",
                "}",
            ]
        )

    if blocking:
        # Synchronous barrier: synchronize on the external stream.
        sync_statement = (
            f"\n{dace_gpu_backend}StreamSynchronize(({dace_gpu_backend}Stream_t){stream_arg});"
        )
        entry_code += sync_statement
        exit_code += sync_statement

    entry_state = sdfg.add_state("sync_entry", is_start_block=True)
    entry_state.add_tasklet(
        "sync_entry_tlet",
        inputs=set(),
        outputs=set(),
        code=entry_code,
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )

    exit_state = sdfg.add_state("sync_exit")
    exit_state.add_tasklet(
        "sync_exit_tlet",
        inputs=set(),
        outputs=set(),
        code=exit_code,
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )

    # NOTE: When synchronization is done through the Tasklet explicitly,
    #   we can disable synchronization for the SDFG state.
    entry_state.nosync = blocking
    exit_state.nosync = blocking

    # Wire the entry state before the original start block and the exit state
    # after all original sink nodes.
    sdfg.add_edge(entry_state, original_start, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(entry_state)

    for sink_node in sdfg.sink_nodes():
        if sink_node is exit_state:
            continue
        sdfg.add_edge(sink_node, exit_state, dace.InterstateEdge())

    # Validate after graph modification.
    sdfg.validate()


def _make_if_region_for_metrics_collection(
    name: str,
    metrics_level: str,
    sdfg: dace.SDFG,
) -> tuple[dace.state.ConditionalBlock, dace.state.SDFGState]:
    """
    Helper function to create a conditional block in the given SDFG, with only one
    branch to be executed if 'metric_level >= metrics.PERFORMANCE' is true.
    """
    if_region = dace.sdfg.state.ConditionalBlock(name)
    sdfg.add_node(if_region, ensure_unique_name=True)
    then_body = dace.sdfg.state.ControlFlowRegion(f"{if_region.label}_collect_metrics", sdfg=sdfg)
    then_state = then_body.add_state(f"{if_region.label}_collect_metrics")
    if_region.add_branch(
        dace.sdfg.state.CodeBlock(f"{metrics_level} >= {metrics.PERFORMANCE}"), then_body
    )
    return if_region, then_state


def add_instrumentation(sdfg: dace.SDFG, gpu: bool) -> None:
    """
    Instrument SDFG with measurement of total execution time.

    We measure the execution time of one GT4Py program by instrumenting the top-level
    SDFG with a cpp timer (std::chrono). This timer measures only the computation
    time, it does not include the overhead of calling the SDFG from Python.

    The execution time is measured in seconds and represented as a 'float64' value.
    It is written to the global array 'SDFG_ARG_METRIC_COMPUTE_TIME'.
    """
    output, _ = sdfg.add_array(gtx_wfdcommon.SDFG_ARG_METRIC_COMPUTE_TIME, [1], dace.float64)
    start_time, _ = sdfg.add_scalar("gt_start_time", dace.int64, transient=True)
    metrics_level = sdfg.add_symbol(
        gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL, gtx_wfdcommon.SDFG_ARG_METRIC_LEVEL_DTYPE
    )

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
        sync_code = "/* The SDFG execution should already be synchronized */"
        has_side_effects = False

    #### 2. Timestamp the SDFG entry point.
    start_block = sdfg.start_block
    entry_if_region, begin_state = _make_if_region_for_metrics_collection(
        "metrics_entry", metrics_level, sdfg
    )
    sdfg.add_edge(entry_if_region, start_block, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(entry_if_region)
    assert sdfg.start_block is entry_if_region

    tlet_start_timer = begin_state.add_tasklet(
        "gt_start_timer",
        inputs={},
        outputs={"time"},
        code=f"""\
{sync_code}
auto now = std::chrono::high_resolution_clock::now();
time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
        """,
        language=dace.dtypes.Language.CPP,
        side_effects=has_side_effects,
    )
    begin_state.add_edge(
        tlet_start_timer,
        "time",
        begin_state.add_access(start_time),
        None,
        dace.Memlet(f"{start_time}[0]"),
    )

    #### 3. Collect the SDFG end timestamp and produce the compute metric.
    exit_if_region, end_state = _make_if_region_for_metrics_collection(
        "metrics_exit", metrics_level, sdfg
    )
    for sink_node in sdfg.sink_nodes():
        if sink_node is exit_if_region:
            continue
        sdfg.add_edge(sink_node, exit_if_region, dace.InterstateEdge())
    assert sdfg.in_degree(exit_if_region) > 0

    # Populate the branch that computes the stencil time metric
    tlet_stop_timer = end_state.add_tasklet(
        "gt_stop_timer",
        inputs={"run_cpp_start_time"},
        outputs={"duration"},
        code=f"""\
{sync_code}
auto now = std::chrono::high_resolution_clock::now();
auto run_cpp_end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
duration = static_cast<double>(run_cpp_end_time - run_cpp_start_time) * 1.e-9;
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
        tlet_stop_timer,
        "duration",
        end_state.add_access(output),
        None,
        dace.Memlet(f"{output}[0]"),
    )

    # Check SDFG validity after applying the above changes.
    # Normally, we do not call `SDFGState.add_tasklet()` directly, instead we call
    #  the wrapper provided by `DataflowBuilder`, that modifies the tasklet connectors
    #  to avoid name conflicts with program symbols. However, this method is not
    #  available here, so we have to call the underlying DaCe function directly.
    #  We now run `validate()` to make sure that no name conflict was introduced.
    sdfg.validate()


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        definitions.CompilableProgramDef,
        stages.ProgramSource[code_specs.SDFGCodeSpec],
    ],
    definitions.TranslationStep[code_specs.SDFGCodeSpec],
):
    device_type: core_defs.DeviceType
    auto_optimize: bool
    auto_optimize_args: dict[str, Any] | None
    async_sdfg_call: bool
    unstructured_horizontal_has_unit_stride: bool
    use_metrics: bool
    max_concurrent_gpu_streams: int

    disable_itir_transforms: bool = False
    disable_field_origin_on_program_arguments: bool = False
    use_max_domain_range_on_unstructured_shift: bool | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent_gpu_streams < 0:
            raise ValueError("max_concurrent_gpu_streams must be >= 0.")

    def generate_sdfg(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dace.SDFG:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            max_concurrent_gpu_streams=self.max_concurrent_gpu_streams,
        ):
            return self._generate_sdfg_without_configuring_dace(*args, **kwargs)

    def _generate_sdfg_without_configuring_dace(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
    ) -> dace.SDFG:
        if not self.disable_itir_transforms:
            ir = itir_transforms.apply_fieldview_transforms(
                ir,
                use_max_domain_range_on_unstructured_shift=self.use_max_domain_range_on_unstructured_shift,
                offset_provider=offset_provider,
            )
        offset_provider_type = common.offset_provider_to_type(offset_provider)
        on_gpu = self.device_type != core_defs.DeviceType.CPU

        sdfg = gtx_dace_lowering.build_sdfg_from_gtir(ir, offset_provider_type, column_axis)

        constant_symbols = find_constant_symbols(
            ir,
            sdfg,
            offset_provider_type,
            self.disable_field_origin_on_program_arguments,
            self.unstructured_horizontal_has_unit_stride,
        )

        if self.auto_optimize:
            auto_optimize_args = {} if self.auto_optimize_args is None else self.auto_optimize_args

            gtx_transformations.gt_auto_optimize(
                sdfg,
                gpu=on_gpu,
                constant_symbols=constant_symbols,
                **auto_optimize_args,
            )
        elif on_gpu:
            # Note that `gt_substitute_compiletime_symbols()` will run `gt_simplify()`
            # at entry, in order to avoid some issue in constant propagatation.
            # Besides, `gt_simplify()` will bring the SDFG into a canonical form
            # that the GPU transformations can handle. This is a workaround for
            # an issue with scalar expressions that are promoted to symbolic expressions
            # and computed on the host (CPU), but the intermediate result is written
            # to a GPU global variable (https://github.com/spcl/dace/issues/1773).
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg, constant_symbols, validate=True
            )
            gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        elif len(constant_symbols) != 0:
            # Target CPU without SDFG transformations, but still replace constant symbols.
            # Replacing the SDFG symbols for field origin in global arrays is strictly
            # required by dace orchestration, which runs the translation stage
            # with `disable_field_origin_on_program_arguments=True`. The program
            # decorator used by dace orchestartion cannot handle field origin.
            # It also requires skipping auto-optimize on the `SDFGConvertible` objects,
            # because it targets the full-application SDFG, so we have to explicitly
            # apply `gt_substitute_compiletime_symbols()` here.
            gtx_transformations.gt_substitute_compiletime_symbols(
                sdfg, constant_symbols, validate=True
            )

        add_synchronization(
            sdfg,
            gpu=on_gpu,
            blocking=(not self.async_sdfg_call),
            n_streams=self.max_concurrent_gpu_streams,
        )

        if self.use_metrics:
            add_instrumentation(sdfg, on_gpu)

        return sdfg

    def __call__(
        self, inp: definitions.CompilableProgramDef
    ) -> stages.ProgramSource[code_specs.SDFGCodeSpec]:
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

        module: stages.ProgramSource[code_specs.SDFGCodeSpec] = stages.ProgramSource(
            entry_point=interface.Function(program.id, program_parameters),
            source_code=gtx_wfdcommon.serialize_sdfg_as_json(sdfg),  # type: ignore[arg-type] # The source code is typed as a `str`, but we assign a JSON dictionary.
            library_deps=tuple(),
            code_spec=code_specs.SDFGCodeSpec(),
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator
