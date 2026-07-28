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
import dataclasses
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
from gt4py.next.program_processors.runners.dace.transformations import (
    auto_optimize as gtx_auto_optimize,
)
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
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = [_make_array_buffer(nbytes=128, address=256)]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)
    program.construct_arguments(alpha=2)

    # Workspace configuration is done exactly once and reused afterwards.
    assert program.sdfg_program.initialize.call_count == 1
    assert program.sdfg_program.get_workspace_sizes.call_count == 1
    assert allocator.allocate.call_count == 1
    allocate_request = allocator.allocate.call_args.args[0]
    assert isinstance(allocate_request, gtx_auto_optimize.AllocationRequest)
    assert allocate_request.nbytes == 128
    assert allocate_request.device == core_defs.DeviceType.CPU
    assert program.sdfg_program.set_workspace.call_count == 1
    assert program.sdfg_program.construct_arguments.call_count == 2

    set_workspace_call = program.sdfg_program.set_workspace.call_args
    assert set_workspace_call.args[0] == dace.StorageType.CPU_Heap
    configured_workspace = set_workspace_call.args[1]
    assert program.external_workspaces[dace.StorageType.CPU_Heap] is configured_workspace


def test_construct_arguments_propagates_allocator_error_for_invalid_size_request():
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = ValueError("invalid workspace size request")
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: -1},
    )

    with pytest.raises(ValueError, match="invalid workspace size request"):
        program.construct_arguments(alpha=1)

    allocator.allocate.assert_called_once()
    assert allocator.allocate.call_args.args[0].nbytes == -1
    assert allocator.allocate.call_args.args[0].device == core_defs.DeviceType.CPU
    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_propagates_allocator_error_for_invalid_storage_request():
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = TypeError("invalid storage type request")
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 16},
    )

    with pytest.raises(TypeError, match="invalid storage type request"):
        program.construct_arguments(alpha=1)

    allocator.allocate.assert_called_once()
    assert allocator.allocate.call_args.args[0].nbytes == 16
    assert allocator.allocate.call_args.args[0].device == core_defs.DeviceType.CPU
    program.sdfg_program.set_workspace.assert_not_called()


def _make_array_buffer(*, nbytes: int, address: int, cuda: bool = False) -> mock.MagicMock:
    """A minimal array-like buffer with a configurable base pointer.

    Exposes ``__array_interface__`` (host) or ``__cuda_array_interface__``
    (device) so it is accepted by ``_validate_external_workspace``; the
    ``data`` tuple carries the address that DaCe's ``array_interface_ptr``
    would hand to the SDFG.
    """
    buffer = mock.MagicMock()
    buffer.nbytes = nbytes
    interface = {"data": (address, False), "shape": (nbytes,), "typestr": "|u1", "version": 3}
    setattr(buffer, "__cuda_array_interface__" if cuda else "__array_interface__", interface)
    return buffer


def test_construct_arguments_rejects_buffer_without_array_interface():
    """The allocator must return something ``set_workspace`` can consume."""
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = ["not-an-array"]  # str exposes no array interface
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 64},
    )

    with pytest.raises(TypeError, match="does not expose `__array_interface__`"):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_rejects_misaligned_buffer():
    """A host buffer whose base pointer is not aligned is rejected."""
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = [_make_array_buffer(nbytes=128, address=100)]  # 100 % 256
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    with pytest.raises(ValueError, match="not aligned to the required 256 bytes"):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_accepts_aligned_buffer():
    """A host buffer whose base pointer is aligned is accepted."""
    allocator = mock.MagicMock()
    workspace = _make_array_buffer(nbytes=128, address=1024)  # 1024 % 256 == 0
    allocator.allocate.side_effect = [workspace]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_called_once()
    assert program.external_workspaces[dace.StorageType.CPU_Heap] is workspace


def test_construct_arguments_rejects_misaligned_gpu_buffer():
    """A device buffer whose base pointer is not aligned is rejected."""
    allocator = mock.MagicMock()
    allocator.allocate.side_effect = [_make_array_buffer(nbytes=128, address=100, cuda=True)]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.GPU_Global: 128},
    )

    with (
        mock.patch.object(
            dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
        ),
        pytest.raises(ValueError, match="not aligned to the required 256 bytes"),
    ):
        program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_not_called()


def test_construct_arguments_skips_alignment_when_data_missing():
    """When the array interface omits ``data`` alignment is trust-based."""
    allocator = mock.MagicMock()
    buffer = mock.MagicMock()
    buffer.nbytes = 64
    # ``data`` is optional on the host array interface.
    buffer.__array_interface__ = {"shape": (64,), "typestr": "|u1", "version": 3}
    allocator.allocate.side_effect = [buffer]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 64},
    )

    # Must not raise even though alignment can't be verified.
    program.construct_arguments(alpha=1)

    program.sdfg_program.set_workspace.assert_called_once()


def test_finalize_calls_deallocate_once_per_storage():
    """``finalize()`` releases each workspace exactly once and is idempotent."""
    allocator = mock.MagicMock()
    workspace = _make_array_buffer(nbytes=128, address=256)
    allocator.allocate.side_effect = [workspace]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={dace.StorageType.CPU_Heap: 128},
    )

    program.construct_arguments(alpha=1)

    program.finalize()
    assert allocator.deallocate.call_count == 1
    assert allocator.deallocate.call_args.args[0] is workspace
    assert program.external_workspaces == {}
    # finalize() is idempotent.
    program.finalize()
    assert allocator.deallocate.call_count == 1


def test_finalize_continues_and_is_idempotent_when_deallocate_fails():
    """If one ``deallocate`` raises, the remaining workspaces are still released.

    A failing buffer must not prevent the others from being deallocated, and
    ``external_workspaces`` must still be cleared so a subsequent ``finalize()``
    (e.g. the pool finalizer) is a no-op rather than re-deallocating the
    buffers that already succeeded.
    """
    allocator = mock.MagicMock()
    workspace_a = _make_array_buffer(nbytes=16, address=256)  # aligned for CPU
    workspace_b = _make_array_buffer(nbytes=32, address=512, cuda=True)  # aligned for GPU
    allocator.allocate.side_effect = [workspace_a, workspace_b]
    program = _make_compiled_program(
        external_memory_allocator=allocator,
        workspace_sizes={
            dace.StorageType.CPU_Heap: 16,
            dace.StorageType.GPU_Global: 32,
        },
    )
    with mock.patch.object(
        dace_wf_compilation.core_defs, "CUPY_DEVICE_TYPE", core_defs.DeviceType.CUDA
    ):
        program.construct_arguments(alpha=1)
    # The first deallocate raises; the second must still be called.
    allocator.deallocate.side_effect = [RuntimeError("boom"), None]

    with pytest.warns(UserWarning, match="Failed to deallocate"):
        program.finalize()

    assert allocator.deallocate.call_count == 2
    assert program.external_workspaces == {}
    # finalize() is idempotent even after partial failures.
    program.finalize()
    assert allocator.deallocate.call_count == 2


def test_finalize_with_no_allocator_is_a_safe_noop():
    """A ``None`` allocator performs no work but still clears workspaces."""
    program = _make_compiled_program(external_memory_allocator=None)

    # finalize() must not raise even though no allocator is configured.
    program.finalize()
    assert program.external_workspaces == {}


# --- Phase 5: allocator pickleability -------------------------------------
#
# ``DaCeCompiler`` is the step that gets pickled when the OTF runner offloads
# compilation to a ``ProcessPoolExecutor`` (``otf/runners.py``), and it carries
# the ``external_memory_allocator``. A non-picklable allocator (closure,
# lambda, local class) must fail fast at construction with
# ``AllocatorNotPicklableError`` rather than silently degrading to in-process
# compilation via a generic runner warning.


@dataclasses.dataclass(frozen=True)
class _ModuleLevelPicklableAllocator:
    """A picklable allocator defined at module scope.

    ``allocate``/``deallocate`` are never called by the tests below; only the
    type's picklability and identity through a round-trip matter. Defined at
    module scope (not inside a test) so ``pickle`` can locate it by qualname.
    Frozen with no fields so two instances are structurally equal, mirroring
    a stateless allocator and the frozenness of ``DaCeCompilationArtifact``.
    """

    def allocate(self, request: gtx_auto_optimize.AllocationRequest):
        raise AssertionError("pickleability tests must not call allocate")

    def deallocate(self, buffer) -> None:
        raise AssertionError("pickleability tests must not call deallocate")


def _make_compiler(allocator=None) -> dace_wf_compilation.DaCeCompiler:
    return dace_wf_compilation.DaCeCompiler(
        bind_func_name="bind",
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        device_type=core_defs.DeviceType.CPU,
        external_memory_allocator=allocator,
    )


def test_dace_compiler_rejects_non_picklable_allocator():
    """An allocator that can not be pickled fails fast at construction."""

    class _LocalAllocator:  # local class -> not picklable by qualname
        def allocate(self, request): ...

        def deallocate(self, buffer) -> None: ...

    with pytest.raises(
        dace_wf_compilation.AllocatorNotPicklableError,
        match="external_memory_allocator .* is not picklable",
    ) as excinfo:
        _make_compiler(allocator=_LocalAllocator())

    # The original pickle error is chained so the user can see *why*.
    assert isinstance(excinfo.value.__cause__, Exception)
    assert "Can't pickle" in str(excinfo.value.__cause__)


def test_dace_compiler_accepts_picklable_allocator():
    """A module-level allocator (and the ``None`` default) pass the gate."""
    # ``None`` default: no probe, no raise.
    _make_compiler(allocator=None)

    # Module-level class: picklable, no raise.
    _make_compiler(allocator=_ModuleLevelPicklableAllocator())


def test_dace_compilation_artifact_pickle_round_trip_with_allocator(tmp_path: pathlib.Path):
    """The allocator round-trips through the pickled compilation artifact.

    The existing ``test_dace_compilation_artifact_pickle_round_trip`` covers the
    no-allocator default; this ensures a real allocator is carried through
    serialization with identity of intent preserved (structural equality,
    since the allocator class defines no per-instance state).
    """
    allocator = _ModuleLevelPicklableAllocator()
    artifact = dace_wf_compilation.DaCeCompilationArtifact(
        library_path=tmp_path / "build" / "libprogram.so",
        sdfg_json="{}",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        bind_func_name="update_sdfg_args",
        device_type=core_defs.DeviceType.CPU,
        external_memory_allocator=allocator,
    )

    restored = pickle.loads(pickle.dumps(artifact))

    assert restored == artifact
    assert isinstance(restored.external_memory_allocator, _ModuleLevelPicklableAllocator)
