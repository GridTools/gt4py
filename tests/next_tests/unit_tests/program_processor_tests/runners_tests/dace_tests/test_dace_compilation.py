# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the compilation stage of the dace backend workflow."""

import contextlib
import unittest.mock as mock

import pytest

dace = pytest.importorskip("dace")

from dace.sdfg import nodes as dace_nodes

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.program_processors.runners.dace.workflow import compilation as dace_wf_compilation


_TX = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
_NONE = dace.dtypes.InstrumentationType.No_Instrumentation


def _add_sequential_map(
    sdfg: dace.SDFG, state: dace.SDFGState, name: str, inp: str, out: str
) -> dace_nodes.MapEntry:
    """Add a default-scheduled mapped tasklet copying `inp` to `out`; return its MapEntry."""
    _, map_entry, _ = state.add_mapped_tasklet(
        name,
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet(f"{inp}[__i]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet(f"{out}[__i]")},
        input_nodes={state.add_access(inp)},
        output_nodes={state.add_access(out)},
        external_edges=True,
    )
    return map_entry


def _make_nested_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.MapEntry]:
    nsdfg = dace.SDFG("inner")
    nstate = nsdfg.add_state("inner_state", is_start_block=True)
    for name in "cd":
        nsdfg.add_array(name, shape=(10,), dtype=dace.float64)
    inner_map_entry = _add_sequential_map(nsdfg, nstate, "inner_map", "c", "d")
    return nsdfg, nstate, inner_map_entry


def _make_sdfg_with_gpu_map() -> dace.SDFG:
    """An SDFG with one GPU-scheduled map and a sequentially-scheduled nested SDFG."""
    sdfg = dace.SDFG("gpu_program")
    state = sdfg.add_state("outer_state", is_start_block=True)
    for name in "ab":
        sdfg.add_array(name, shape=(10,), dtype=dace.float64)
    gpu_map_entry = _add_sequential_map(sdfg, state, "gpu_map", "a", "b")
    gpu_map_entry.map.schedule = dace.dtypes.ScheduleType.GPU_Device
    _add_sequential_map(sdfg, state, "seq_map", "a", "b")
    nsdfg, _, _ = _make_nested_sdfg()
    state.add_nested_sdfg(nsdfg, inputs={"c"}, outputs={"d"})
    return sdfg


def _run_compiler(
    tmp_path, *, add_gpu_trace_markers: bool, device_type: core_defs.DeviceType
) -> tuple[mock.MagicMock, dace.SDFG]:
    """Run `DaCeCompiler` on a GPU SDFG with compilation stubbed out.

    Returns the spy wrapping `_add_tx_markers` and the SDFG that was handed to
    ``SDFG.compile`` (i.e. the SDFG after any marker processing).
    """
    inp = mock.MagicMock()
    inp.program_source.source_code = _make_sdfg_with_gpu_map().to_json()

    compiler = dace_wf_compilation.DaCeCompiler(
        bind_func_name="bind",
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        device_type=device_type,
        add_gpu_trace_markers=add_gpu_trace_markers,
    )

    with (
        mock.patch.object(
            dace_wf_compilation,
            "_add_tx_markers",
            wraps=dace_wf_compilation._add_tx_markers,
        ) as spy,
        mock.patch.object(dace.SDFG, "compile", autospec=True) as compile_mock,
        mock.patch.object(dace_wf_compilation, "CompiledDaceProgram"),
        mock.patch.object(
            dace_wf_compilation.gtx_wfdcommon,
            "dace_context",
            lambda **kwargs: contextlib.nullcontext(),
        ),
        mock.patch.object(dace_wf_compilation.gtx_cache, "get_cache_folder", return_value=tmp_path),
        mock.patch.object(
            dace_wf_compilation.locking, "lock", lambda *args, **kwargs: contextlib.nullcontext()
        ),
        # Pretend cupy/CUDA is available so the `device_type == CUPY_DEVICE_TYPE` guard
        # can be exercised on a CPU-only machine.
        mock.patch.object(
            dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
        ),
    ):
        compiler(inp)
        compiled_sdfg = compile_mock.call_args.args[0]

    return spy, compiled_sdfg


def test_compiler_applies_tx_markers_for_gpu(tmp_path):
    """On a CUDA target with the flag on, the compiler applies the markers to the SDFG."""
    spy, compiled_sdfg = _run_compiler(
        tmp_path, add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CUDA
    )

    spy.assert_called_once()
    # The SDFG that was marked is the very one passed on to compilation.
    assert spy.call_args.args[0] is compiled_sdfg
    assert compiled_sdfg.instrument == _TX
    map_entries = [
        n for n, _ in compiled_sdfg.all_nodes_recursive() if isinstance(n, dace_nodes.MapEntry)
    ]
    assert map_entries and all(me.instrument == _TX for me in map_entries)


def test_compiler_skips_tx_markers_when_flag_disabled(tmp_path):
    """With the flag off the compiler must not touch instrumentation, even on CUDA."""
    spy, compiled_sdfg = _run_compiler(
        tmp_path, add_gpu_trace_markers=False, device_type=core_defs.DeviceType.CUDA
    )

    spy.assert_not_called()
    assert compiled_sdfg.instrument == _NONE


def test_compiler_skips_tx_markers_for_non_gpu_device(tmp_path):
    """On a CPU target the markers must not be applied even with the flag on."""
    spy, compiled_sdfg = _run_compiler(
        tmp_path, add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CPU
    )

    spy.assert_not_called()
    assert compiled_sdfg.instrument == _NONE
