# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compilation stage of the dace backend workflow.

Covers the GPU TX-marker instrumentation and the picklability of
``DaCeCompilationArtifact``.
"""

import contextlib
import pathlib
import pickle
import unittest.mock as mock
from typing import Any

import numpy as np
import pytest

dace = pytest.importorskip("dace")

from dace.sdfg import nodes as dace_nodes

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import code_specs, stages
from gt4py.next.otf.binding import interface
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
    nested_node = state.add_nested_sdfg(nsdfg, inputs={"c"}, outputs={"d"})
    a_in = state.add_access("a")
    b_out = state.add_access("b")
    state.add_edge(a_in, None, nested_node, "c", dace.Memlet("a[0:10]"))
    state.add_edge(nested_node, "d", b_out, None, dace.Memlet("b[0:10]"))
    sdfg.validate()
    return sdfg


def _make_extension_source() -> stages.ExtensionSource:
    """A real `ExtensionSource` wrapping the GPU SDFG, as the dace translation step emits.

    Using a real source (rather than a `MagicMock`) lets the unmocked `get_cache_folder`
    fingerprint the program source for the build-folder name.
    """
    program_source = stages.ProgramSource(
        entry_point=interface.Function("gpu_program", parameters=()),
        source_code=_make_sdfg_with_gpu_map().to_json(),
        library_deps=(),
        code_spec=code_specs.SDFGCodeSpec(),
    )
    binding_source = stages.BindingSource(source_code="", library_deps=())
    return stages.ExtensionSource(program_source=program_source, binding_source=binding_source)


def _run_compiler(
    *,
    add_gpu_trace_markers: bool = False,
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.RELEASE,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> tuple[dace_wf_compilation.DaCeCompilationArtifact, dace.SDFG, mock.MagicMock]:
    """Run `DaCeCompiler` on a GPU SDFG with compilation stubbed out.

    Returns the compilation artifact, the SDFG which was compiled, and the spy wrapping `_add_tx_markers`.
    """
    inp = _make_extension_source()

    compiler = dace_wf_compilation.DaCeCompiler(
        bind_func_name="bind",
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        device_type=device_type,
        add_gpu_trace_markers=add_gpu_trace_markers,
        cmake_build_type=cmake_build_type,
    )

    with (
        mock.patch.object(
            dace_wf_compilation,
            "_add_tx_markers",
            wraps=dace_wf_compilation._add_tx_markers,
        ) as spy_add_tx_markers,
        mock.patch.object(dace.SDFG, "compile", autospec=True) as compile_mock,
        mock.patch.object(
            dace_wf_compilation.locking, "lock", lambda *args, **kwargs: contextlib.nullcontext()
        ),
        # Pretend cupy/CUDA is available so the `device_type == CUPY_DEVICE_TYPE` guard
        # can be exercised on a CPU-only machine.
        mock.patch.object(
            dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
        ),
        mock.patch("gt4py.next.otf.compilation.common.get_device_arch", return_value="xyz"),
    ):
        artifact = compiler(inp)
        compile_input = compile_mock.call_args.args[0]

    return artifact, compile_input, spy_add_tx_markers


def test_compiler_applies_tx_markers_for_gpu():
    """On a CUDA target with the flag on, the compiler applies the markers to the SDFG."""
    _, compile_input, spy = _run_compiler(
        add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CUDA
    )

    spy.assert_called_once()
    # The SDFG that was marked is the very one passed on to compilation.
    assert spy.call_args.args[0] is compile_input
    assert compile_input.instrument == _TX
    map_entries = [
        n for n, _ in compile_input.all_nodes_recursive() if isinstance(n, dace_nodes.MapEntry)
    ]
    assert map_entries and all(me.instrument == _TX for me in map_entries)


def test_compiler_skips_tx_markers_when_flag_disabled():
    """With the flag off the compiler must not touch instrumentation, even on CUDA."""
    _, compile_input, spy = _run_compiler(
        add_gpu_trace_markers=False, device_type=core_defs.DeviceType.CUDA
    )

    spy.assert_not_called()
    assert compile_input.instrument == _NONE


def test_compiler_skips_tx_markers_for_non_gpu_device():
    """On a CPU target the markers must not be applied even with the flag on."""
    _, compile_input, spy = _run_compiler(
        add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CPU
    )

    spy.assert_not_called()
    assert compile_input.instrument == _NONE


def test_dace_compilation_artifact_pickle_round_trip(tmp_path: pathlib.Path):
    artifact = dace_wf_compilation.DaCeCompilationArtifact(
        library_path=tmp_path / "build" / "libprogram.so",
        sdfg_json="{}",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        bind_func_name="update_sdfg_args",
        device_type=core_defs.DeviceType.CPU,
    )

    restored = pickle.loads(pickle.dumps(artifact))

    assert restored == artifact


# `CXXFLAGS`, `CUDAFLAGS` and `HIPFLAGS` feed `compiler.cpu.args`, `compiler.cuda.args`
# and `compiler.cuda.hip_args` respectively (see `set_dace_config`).
@pytest.mark.parametrize(
    ("device_type", "compiler_flags_env"),
    [
        (core_defs.DeviceType.CPU, "CXXFLAGS"),
        (core_defs.DeviceType.CUDA, "CUDAFLAGS"),
        (core_defs.DeviceType.ROCM, "HIPFLAGS"),
    ],
)
def test_compiler_flags_change_build_folder(monkeypatch, device_type, compiler_flags_env):
    """Different compiler flags must produce a different build folder.

    The flags are captured in `dace_config_nondefaults`, whose fingerprint the compiler
    passes to `get_cache_folder` as the `build_context_id`. That id is appended to the
    build-folder name, so changing any flag lands the build in a different folder of the
    build cache.
    """
    monkeypatch.delenv(compiler_flags_env, raising=False)
    artifact_default, _, _ = _run_compiler(device_type=device_type)

    monkeypatch.setenv(compiler_flags_env, "-O0 -some-custom-flag")
    artifact_custom, _, _ = _run_compiler(device_type=device_type)

    # The differing `dace_config_nondefaults` make the two compilers fingerprint differently,
    # so `get_cache_folder` names two distinct build folders.
    assert artifact_default.library_path != artifact_custom.library_path


def test_cmake_build_type_changes_build_folder():
    """Different cmake build types must produce a different SDFG build folder.

    The cmake build type is part of the DaCe configuration captured in
    `dace_config_nondefaults`, whose fingerprint is passed to `get_cache_folder`
    as `build_context_id`. That id is appended to the build-folder name, so
    changing the build type lands the SDFG build in a different folder of the
    build cache.
    """
    artifact_release, _, _ = _run_compiler(cmake_build_type=config.CMakeBuildType.RELEASE)
    artifact_debug, _, _ = _run_compiler(cmake_build_type=config.CMakeBuildType.DEBUG)

    assert artifact_release.library_path != artifact_debug.library_path


def _make_compiled_program(
    *,
    external_memory_allocator=None,
    workspace_sizes: dict[Any, int] | None = None,
):
    if workspace_sizes is None:
        workspace_sizes = {}

    sdfg = mock.MagicMock()
    sdfg.arglist.return_value = {}
    sdfg.build_folder = "build-folder"

    sdfg_program = mock.MagicMock()
    sdfg_program.sdfg = sdfg
    sdfg_program.get_workspace_sizes.return_value = workspace_sizes
    sdfg_program.construct_arguments.return_value = ((), ())

    return dace_wf_compilation.CompiledDaceProgram(
        program=sdfg_program,
        bind_func_name="update_sdfg_args",
        binding_source_code="def update_sdfg_args(*a, **k):\n    return None\n",
        external_memory_allocator=external_memory_allocator,
    )


def test_construct_arguments_installs_external_workspaces_once():
    allocator = mock.MagicMock(side_effect=[np.empty((128,), dtype=np.uint8)])
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)
    program.construct_arguments(alpha=2)

    # Workspace configuration is done exactly once and reused afterwards.
    assert program.sdfg_program.initialize.call_count == 1
    assert program.sdfg_program.get_workspace_sizes.call_count == 1
    assert allocator.call_count == 1
    allocator.assert_called_once_with(128, core_defs.DeviceType.CPU)
    assert program.sdfg_program.set_workspace.call_count == 1
    assert program.sdfg_program.construct_arguments.call_count == 2

    set_workspace_call = program.sdfg_program.set_workspace.call_args
    assert set_workspace_call.args[0] == dace.StorageType.CPU_Heap
    configured_workspace = set_workspace_call.args[1]
    assert program.external_workspaces[dace.StorageType.CPU_Heap] is configured_workspace


def test_construct_arguments_propagates_allocator_error_for_invalid_size_request():
    allocator = mock.MagicMock(side_effect=ValueError("invalid workspace size request"))
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: -1},
    )

    with pytest.raises(ValueError, match="invalid workspace size request"):
        program.construct_arguments(alpha=1)

    allocator.assert_called_once_with(-1, core_defs.DeviceType.CPU)
    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_propagates_allocator_error_for_invalid_storage_request():
    allocator = mock.MagicMock(side_effect=TypeError("invalid storage type request"))
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 16},
    )

    with pytest.raises(TypeError, match="invalid storage type request"):
        program.construct_arguments(alpha=1)

    allocator.assert_called_once_with(16, core_defs.DeviceType.CPU)
    program.sdfg_program.set_workspace.assert_not_called()
