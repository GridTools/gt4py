# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compilation stage of the dace backend workflow.

Covers the GPU TX-marker instrumentation and external workspace handling of
``CompiledDaceProgram`` / ``DaCeCompilationArtifact``.
"""

import contextlib
import pathlib
import pickle
import unittest.mock as mock
from typing import Any

import pytest

dace = pytest.importorskip("dace")
import dace.codegen.compiler as dace_compiler

from dace.sdfg import nodes as dace_nodes

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import code_specs, stages
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners.dace.workflow import (
    compilation as dace_wf_compilation,
)
from gt4py.next.program_processors.runners.dace.workflow import (
    common as dace_wf_common,
    compilation as dace_wf_compilation,
)


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


@pytest.fixture
def program_source() -> dace_wf_compilation.SDFGExtensionSource:
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
    inp: stages.ExtensionSource,
    *,
    add_gpu_trace_markers: bool = False,
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.RELEASE,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
) -> tuple[dace_wf_compilation.DaCeCompilationArtifact, dace.SDFG]:
    """Run `DaCeCompiler` on the provided `program_source` with compilation stubbed out.

    Returns the compilation artifact and the SDFG which was compiled.
    """
    compiler = dace_wf_compilation.DaCeCompiler(
        bind_func_name="bind",
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        device_type=device_type,
        add_gpu_trace_markers=add_gpu_trace_markers,
        cmake_build_type=cmake_build_type,
    )

    with (
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

    return artifact, compile_input


def test_compiler_applies_tx_markers_for_gpu(program_source):
    """On a CUDA target with the flag on, the compiler applies the markers to the SDFG."""
    _, compile_input = _run_compiler(
        program_source, add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CUDA
    )

    assert compile_input.instrument == _TX
    map_entries = [
        n for n, _ in compile_input.all_nodes_recursive() if isinstance(n, dace_nodes.MapEntry)
    ]
    assert map_entries and all(me.instrument == _TX for me in map_entries)


def test_compiler_skips_tx_markers_when_flag_disabled(program_source):
    """With the flag off the compiler must not touch instrumentation, even on CUDA."""
    _, compile_input = _run_compiler(
        program_source, add_gpu_trace_markers=False, device_type=core_defs.DeviceType.CUDA
    )

    assert compile_input.instrument == _NONE


def test_compiler_skips_tx_markers_for_non_gpu_device(program_source):
    """On a CPU target the markers must not be applied even with the flag on."""
    _, compile_input = _run_compiler(
        program_source, add_gpu_trace_markers=True, device_type=core_defs.DeviceType.CPU
    )

    assert compile_input.instrument == _NONE


def test_dace_compilation_artifact_pickle_round_trip(tmp_path: pathlib.Path):
    """The artifact is picklable and does not carry runtime workspace buffers."""
    artifact = dace_wf_compilation.DaCeCompilationArtifact(
        library_path=tmp_path / "build" / "libprogram.so",
        sdfg_json="{}",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        bind_func_name="update_sdfg_args",
        device_type=core_defs.DeviceType.CPU,
    )

    restored = pickle.loads(pickle.dumps(artifact))

    assert restored == artifact


@pytest.mark.parametrize("add_gpu_trace_markers", [False, True])
def test_same_artifact(add_gpu_trace_markers, program_source):
    """Same SDFG and compile settings must produce the same artifact.

    We also test the case ``add_gpu_trace_markers=True`` to verify that the modified
    SDFG has the same fingerprint, for the same input SDFG and compilation settings.
    This way, we verify that the GUIDs elements are removed from the JSON source code.
    """
    artifact_1, sdfg_1 = _run_compiler(
        program_source,
        add_gpu_trace_markers=add_gpu_trace_markers,
        device_type=core_defs.DeviceType.CUDA,
    )
    artifact_2, sdfg_2 = _run_compiler(
        program_source,
        add_gpu_trace_markers=add_gpu_trace_markers,
        device_type=core_defs.DeviceType.CUDA,
    )

    assert artifact_1.library_path == artifact_2.library_path
    assert (
        sdfg_1.hash_sdfg() == sdfg_2.hash_sdfg()
    )  # might contain different GUIDs, `hash_sdfg()` ignores them


def test_apply_tx_markers_changes_artifact(program_source):
    """Different instrumentation settings must produce a different artifact."""
    artifact_base, _ = _run_compiler(
        program_source, device_type=core_defs.DeviceType.CUDA, add_gpu_trace_markers=False
    )
    artifact_with_markers, _ = _run_compiler(
        program_source, device_type=core_defs.DeviceType.CUDA, add_gpu_trace_markers=True
    )

    assert artifact_base.library_path != artifact_with_markers.library_path


# `CXXFLAGS`, `CUDAFLAGS` and `HIPFLAGS` feed `compiler.cpu.args`, `compiler.cuda.args`
# and `compiler.cuda.hip_args` respectively (see `set_dace_config`).
@pytest.mark.parametrize(
    ("device_type", "compiler_flags_env"),
    [
        (core_defs.DeviceType.CPU, "CXXFLAGS"),
        (core_defs.DeviceType.CUDA, "CUDAFLAGS"),
        (core_defs.DeviceType.ROCM, "HIPFLAGS"),
    ],
    ids=["CPU", "CUDA", "HIP"],
)
def test_compiler_flags_change_artifact(
    device_type, compiler_flags_env, program_source, monkeypatch
):
    """Different compiler flags must produce a different artifact.

    The flags are captured in `dace_config_nondefaults`, whose fingerprint the compiler
    passes to `get_cache_folder` as the `build_context_id`. That id is appended to the
    build-folder name, so changing any flag lands the build in a different folder of the
    build cache.
    """
    monkeypatch.delenv(compiler_flags_env, raising=False)
    artifact_default, _ = _run_compiler(program_source, device_type=device_type)

    monkeypatch.setenv(compiler_flags_env, "-O0 -some-custom-flag")
    artifact_custom, _ = _run_compiler(program_source, device_type=device_type)

    # The differing `dace_config_nondefaults` make the two compilers fingerprint differently,
    # so `get_cache_folder` names two distinct artifacts.
    assert artifact_default.library_path != artifact_custom.library_path


def test_cmake_build_type_changes_artifact(program_source):
    """Different cmake build types must produce a different SDFG artifact.

    The cmake build type is part of the DaCe configuration captured in
    `dace_config_nondefaults`, whose fingerprint is passed to `get_cache_folder`
    as `build_context_id`. That id is appended to the build-folder name, so
    changing the build type lands the SDFG build in a different folder of the
    build cache.
    """
    artifact_release, _ = _run_compiler(
        program_source, cmake_build_type=config.CMakeBuildType.RELEASE
    )
    artifact_debug, _ = _run_compiler(program_source, cmake_build_type=config.CMakeBuildType.DEBUG)

    assert artifact_release.library_path != artifact_debug.library_path


def _make_compiled_program(
    *,
    external_workspace: dict[core_defs.DeviceType, Any] | None = None,
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

    compiled_program = dace_wf_compilation.CompiledDaceProgram(
        program=sdfg_program,
        bind_func_name="update_sdfg_args",
        binding_source_code="def update_sdfg_args(*a, **k):\n    return None\n",
    )
    if external_workspace is not None:
        compiled_program.external_workspace = external_workspace
    return compiled_program


def test_construct_arguments_without_external_workspace():
    """If the SDFG does not need a workspace, `construct_arguments` works without it."""
    program = _make_compiled_program(
        external_workspace=None, workspace_sizes={}
    )  # no workspace needed

    program.construct_arguments(alpha=1)

    program.sdfg_program.initialize.assert_called_once()
    program.sdfg_program.get_workspace_sizes.assert_called_once()
    program.sdfg_program.set_workspace.assert_not_called()
    program.sdfg_program.construct_arguments.assert_called_once()


def test_construct_arguments_installs_external_workspace():
    """If the SDFG needs a workspace, `construct_arguments` installs it from the mapping."""
    workspace = _make_array_buffer(nbytes=128)
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: workspace},
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)

    assert program.sdfg_program.initialize.call_count == 1
    assert program.sdfg_program.get_workspace_sizes.call_count == 1

    set_workspace_call = program.sdfg_program.set_workspace.call_args
    assert set_workspace_call.args[0] == dace.StorageType.CPU_Heap
    assert set_workspace_call.args[1] is workspace


def test_construct_arguments_raises_when_workspace_missing():
    """If the SDFG needs a workspace but none is provided, raise early."""
    program = _make_compiled_program(
        external_workspace=None, workspace_sizes={dace.StorageType.CPU_Heap: 128}
    )

    with pytest.raises(RuntimeError, match="External workspace is not set"):
        program.construct_arguments(alpha=1)


def test_construct_arguments_raises_when_workspace_missing_for_device():
    """If a required device workspace is absent from the mapping, raise early."""
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: _make_array_buffer(nbytes=128)},
        workspace_sizes={dace.StorageType.GPU_Global: 128},
    )
    with mock.patch.object(
        dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
    ):
        with pytest.raises(RuntimeError, match="External workspace for device .* not found"):
            program.construct_arguments(alpha=1)


def test_construct_arguments_propagates_validation_error_for_too_small_buffer():
    """A workspace that is too small for the requested storage is rejected."""
    workspace = _make_array_buffer(nbytes=64)
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: workspace},
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    with pytest.raises(ValueError, match="at least 128 bytes were required"):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_propagates_validation_error_for_invalid_storage():
    """An unsupported storage type is rejected during device mapping."""
    workspace = _make_array_buffer(nbytes=128)
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: workspace},
        workspace_sizes={dace.StorageType.CPU_Pinned: 128},
    )

    with pytest.raises(ValueError, match="Unsupported storage type"):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def _make_array_buffer(*, nbytes: int, cuda: bool = False) -> mock.MagicMock:
    """A minimal array-like buffer accepted by ``_validate_external_workspace``.

    Exposes ``__array_interface__`` (host) or ``__cuda_array_interface__``
    (device); ``nbytes`` matches the requested size.
    """
    buffer = mock.MagicMock()
    buffer.nbytes = nbytes
    interface = {"shape": (nbytes,), "typestr": "|u1", "version": 3}
    setattr(buffer, "__cuda_array_interface__" if cuda else "__array_interface__", interface)
    return buffer


def test_construct_arguments_rejects_buffer_without_array_interface():
    """The workspace must expose an array interface that ``set_workspace`` can consume."""
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: "not-an-array"},
        workspace_sizes={dace.StorageType.CPU_Heap: 64},
    )

    with pytest.raises(TypeError, match="must expose `__array_interface__`"):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_accepts_cpu_workspace():
    """A host workspace with matching size is accepted and installed."""
    workspace = _make_array_buffer(nbytes=128)
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: workspace},
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_called_once()
    assert program.sdfg_program.set_workspace.call_args.args[1] is workspace


def test_construct_arguments_accepts_gpu_workspace():
    """A device workspace with matching size is accepted and installed."""
    workspace = _make_array_buffer(nbytes=128, cuda=True)
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CUDA: workspace},
        workspace_sizes={dace.StorageType.GPU_Global: 128},
    )
    with mock.patch.object(
        dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
    ):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_called_once()
    assert program.sdfg_program.set_workspace.call_args.args[0] == dace.StorageType.GPU_Global
    assert program.sdfg_program.set_workspace.call_args.args[1] is workspace


class _ArrayBufferWithoutNbytes:
    __array_interface__ = {"shape": (128,), "typestr": "|u1", "version": 3}


def test_construct_arguments_skips_size_check_when_nbytes_missing():
    """When the buffer lacks ``nbytes`` the size contract is trust-based."""
    program = _make_compiled_program(
        external_workspace={core_defs.DeviceType.CPU: _ArrayBufferWithoutNbytes()},
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    # Must not raise even though the size cannot be verified.
    program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_called_once()


class _FakeStream:
    """Stand-in for a ``cupy.cuda.Stream`` that does not require a GPU."""

    def __init__(self, ptr: int = 42, device_id: int = 0) -> None:
        self.ptr = ptr
        self.device_id = device_id


class _FakeEvent:
    """Stand-in for a ``cupy.cuda.Event`` that does not require a GPU."""

    def __init__(self, ptr: int = 99) -> None:
        self.ptr = ptr


def _make_sdfg_with_stream_sync_symbols() -> dace.SDFG:
    """Return a minimal SDFG whose arglist contains the stream-sync symbols."""
    sdfg = dace.SDFG("stream_sync_program")
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol(dace_wf_common.SDFG_ARG_EXTERNAL_WS_EVENT, dace.uint64)
    sdfg.add_symbol(dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM, dace.uint64)
    # Add a no-op tasklet so the SDFG is not empty.
    state.add_tasklet(
        "noop", inputs=set(), outputs=set(), code="", language=dace.dtypes.Language.CPP
    )
    sdfg.validate()
    return sdfg


def test_compiled_program_creates_event_when_sdfg_has_sync_symbols(monkeypatch):
    """If the SDFG contains the event symbol, CompiledDaceProgram creates a cupy Event."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    fake_cupy_module.cuda.Event.return_value = _FakeEvent(123)
    fake_cupy_module.cuda.Device.return_value.id = 0
    fake_cupy_module.cuda.runtime.cudaSuccess = 0
    fake_cupy_module.cuda.runtime.cudaErrorNotReady = 600
    fake_cupy_module.cuda.runtime.cudaStreamQuery.return_value = 0
    monkeypatch.setattr(dace_wf_compilation, "cupy", fake_cupy_module)

    compiled_sdfg = mock.MagicMock()
    compiled_sdfg.sdfg = _make_sdfg_with_stream_sync_symbols()
    compiled_sdfg.sdfg.build_folder = "/tmp"

    program = dace_wf_compilation.CompiledDaceProgram(
        compiled_sdfg,
        bind_func_name="update_sdfg_args",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        device_type=core_defs.DeviceType.CUDA,
        external_sync_stream=_FakeStream(ptr=7, device_id=0),
    )

    assert program._sync_event is not None
    assert program._sync_event.ptr == 123
    assert program.external_sync_stream.ptr == 7


def test_compiled_program_adds_stream_sync_kwargs(monkeypatch):
    """construct_arguments forwards event/stream pointer values to the SDFG."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    fake_cupy_module.cuda.Event.return_value = _FakeEvent(123)
    fake_cupy_module.cuda.Device.return_value.id = 0
    fake_cupy_module.cuda.runtime.cudaSuccess = 0
    fake_cupy_module.cuda.runtime.cudaErrorNotReady = 600
    fake_cupy_module.cuda.runtime.cudaStreamQuery.return_value = 0
    monkeypatch.setattr(dace_wf_compilation, "cupy", fake_cupy_module)

    compiled_sdfg = mock.MagicMock()
    compiled_sdfg.sdfg = _make_sdfg_with_stream_sync_symbols()
    compiled_sdfg.sdfg.build_folder = "/tmp"
    compiled_sdfg.construct_arguments.return_value = ([], [])

    program = dace_wf_compilation.CompiledDaceProgram(
        compiled_sdfg,
        bind_func_name="update_sdfg_args",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        device_type=core_defs.DeviceType.CUDA,
        external_sync_stream=_FakeStream(ptr=7, device_id=0),
    )

    with mock.patch.object(program, "_configure_external_workspaces"):
        program.construct_arguments(some_arg=1)

    call_kwargs = compiled_sdfg.construct_arguments.call_args.kwargs
    assert call_kwargs[dace_wf_common.SDFG_ARG_EXTERNAL_WS_EVENT] == 123
    assert call_kwargs[dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM] == 7


def test_compiled_program_rejects_non_cupy_stream(monkeypatch):
    """A non-cupy stream object raises TypeError at construction time."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    monkeypatch.setattr(dace_wf_compilation, "cupy", fake_cupy_module)

    compiled_sdfg = mock.MagicMock()
    compiled_sdfg.sdfg = _make_sdfg_with_stream_sync_symbols()
    compiled_sdfg.sdfg.build_folder = "/tmp"

    with pytest.raises(TypeError, match="external_sync_stream must be a cupy.cuda.Stream"):
        dace_wf_compilation.CompiledDaceProgram(
            compiled_sdfg,
            bind_func_name="update_sdfg_args",
            binding_source_code="def update_sdfg_args(*a, **k): ...",
            device_type=core_defs.DeviceType.CUDA,
            external_sync_stream="not-a-stream",
        )


def test_compiled_program_rejects_cpu_with_stream_sync_symbols(monkeypatch):
    """A CPU target with stream-sync symbols requires an external stream (and would still fail)."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    fake_cupy_module.cuda.Event.return_value = _FakeEvent(123)
    monkeypatch.setattr(dace_wf_compilation, "cupy", fake_cupy_module)

    compiled_sdfg = mock.MagicMock()
    compiled_sdfg.sdfg = _make_sdfg_with_stream_sync_symbols()
    compiled_sdfg.sdfg.build_folder = "/tmp"

    with pytest.raises(ValueError, match="external_sync_stream is not supported on CPU targets"):
        dace_wf_compilation.CompiledDaceProgram(
            compiled_sdfg,
            bind_func_name="update_sdfg_args",
            binding_source_code="def update_sdfg_args(*a, **k): ...",
            device_type=core_defs.DeviceType.CPU,
            external_sync_stream=_FakeStream(),
        )


def test_compilation_artifact_load_passes_external_sync_stream(monkeypatch, tmp_path: pathlib.Path):
    """DaCeBackend.compile passes the stream through artifact.load() to CompiledDaceProgram."""
    fake_stream = _FakeStream()
    mock_program = mock.MagicMock()
    monkeypatch.setattr(
        dace_wf_compilation,
        "CompiledDaceProgram",
        mock.MagicMock(return_value=mock_program),
    )

    artifact = dace_wf_compilation.DaCeCompilationArtifact(
        library_path=tmp_path / "build" / "libprogram.so",
        sdfg_json="{}",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        bind_func_name="update_sdfg_args",
        device_type=core_defs.DeviceType.CUDA,
    )

    with mock.patch.object(dace.SDFG, "from_json", return_value=dace.SDFG("test")):
        with mock.patch.object(dace_compiler, "get_program_handle", return_value=mock.MagicMock()):
            artifact.load(external_sync_stream=fake_stream)

    compiled_dace_program_call = dace_wf_compilation.CompiledDaceProgram.call_args
    assert compiled_dace_program_call.kwargs["external_sync_stream"] is fake_stream
