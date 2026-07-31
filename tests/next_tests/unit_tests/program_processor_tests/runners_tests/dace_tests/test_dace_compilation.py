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
        sdfg_build_folder=tmp_path,
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

    assert artifact_1.sdfg_build_folder == artifact_2.sdfg_build_folder
    assert artifact_1 == artifact_2
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

    assert artifact_base.sdfg_build_folder != artifact_with_markers.sdfg_build_folder


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
    assert artifact_default.sdfg_build_folder != artifact_custom.sdfg_build_folder


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

    assert artifact_release.sdfg_build_folder != artifact_debug.sdfg_build_folder
