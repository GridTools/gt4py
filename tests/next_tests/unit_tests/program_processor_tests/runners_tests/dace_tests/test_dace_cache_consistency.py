# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Crash- and concurrency-consistency of the dace build-folder cache.

With ``compiler.use_cache=True`` (set by the dace workflow) dace reuses a build
folder whenever the compiled library merely *exists* and never validates it. An
``sdfg.compile()`` interrupted mid-link leaves a truncated, unloadable
``lib<name>.so`` behind; the next run accepts the HIT and every subsequent
``load()`` crashes on ``dlopen`` until the folder is cleaned manually.

The compile step therefore records a completion marker after each successful
compile; a library without the marker is treated as stale and dropped, forcing
a rebuild.

Concurrent compiles of one program serialize on a lock over the build folder,
and the compile step does not read the folder before it holds that lock.
"""

import contextlib
import ctypes
import pathlib
import shutil

import dace
import pytest

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config, fingerprinting
from gt4py.next.otf import artifacts
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import compilation as dace_wf_compilation


def _make_compilable_sdfg(name: str) -> dace.SDFG:
    """A minimal CPU-compilable SDFG: copy ``a`` to ``b`` over a sequential map."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state("state", is_start_block=True)
    for array_name in ("a", "b"):
        sdfg.add_array(array_name, shape=(10,), dtype=dace.float64)
    state.add_mapped_tasklet(
        "copy",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={state.add_access("a")},
        output_nodes={state.add_access("b")},
        external_edges=True,
    )
    return sdfg


def _make_input(name: str) -> artifacts.ExtensionSource:
    sdfg = _make_compilable_sdfg(name)
    program_source = artifacts.ProgramSource(
        entry_point=interface.Function(name=sdfg.name, parameters=()),
        source_code=sdfg.to_json(),
        library_deps=(),
        code_spec=artifacts.SDFGCodeSpec(),
    )
    return artifacts.ExtensionSource(
        program_source=program_source,
        binding_source=artifacts.BindingSource(
            source_code="def bind(*args, **kwargs):\n    return None\n",
            library_deps=(),
        ),
    )


def _compiler() -> dace_wf_compilation.DaCeCompiler:
    return dace_wf_compilation.DaCeCompiler(
        bind_func_name="bind",
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        device_type=core_defs.DeviceType.CPU,
        add_gpu_trace_markers=False,
    )


def _build_folder(
    comp: dace_wf_compilation.DaCeCompiler, inp: artifacts.ExtensionSource
) -> pathlib.Path:
    return gtx_cache.get_cache_folder(
        inp,
        config.BuildCacheLifetime.SESSION,
        build_context_id=fingerprinting.strict_fingerprinter(comp.dace_config_nondefaults),
    )


@pytest.fixture
def clean_build_folder(request):
    def factory(
        comp: dace_wf_compilation.DaCeCompiler, inp: artifacts.ExtensionSource
    ) -> pathlib.Path:
        folder = _build_folder(comp, inp)
        shutil.rmtree(folder, ignore_errors=True)
        request.addfinalizer(lambda: shutil.rmtree(folder, ignore_errors=True))
        return folder

    return factory


def test_dace_recovers_from_truncated_library(clean_build_folder):
    """A truncated (but present) ``lib<name>.so`` is accepted by dace's
    existence-only cache hit. The compile step must detect the incomplete build
    (missing completion marker) and rebuild instead of handing out a library
    that fails to ``dlopen``."""
    inp = _make_input("truncated_library")
    comp = _compiler()
    build_folder = clean_build_folder(comp, inp)

    artifact = comp(inp)
    assert artifact.library_path.is_file()

    # Simulate a build interrupted mid-link: truncated library, no completion marker.
    artifact.library_path.write_bytes(b"\x00" * 64)
    (build_folder / dace_wf_compilation._COMPILE_COMPLETE_MARKER).unlink()

    recovered = comp(inp)

    ctypes.CDLL(str(recovered.library_path))  # raises OSError if still truncated


def test_dace_recovers_from_empty_folder_mode(clean_build_folder):
    """dace creates ``FOLDER_MODE`` before it writes the mode into it, so a build
    killed in between leaves an empty marker, which dace reads as the unknown mode
    ``''`` and refuses. The compile step must detect the incomplete build (missing
    completion marker) and rebuild instead of failing on that folder for good."""
    inp = _make_input("empty_folder_mode")
    comp = _compiler()
    build_folder = clean_build_folder(comp, inp)

    comp(inp)

    (build_folder / "FOLDER_MODE").write_text("")
    (build_folder / dace_wf_compilation._COMPILE_COMPLETE_MARKER).unlink()

    recovered = comp(inp)

    ctypes.CDLL(str(recovered.library_path))


def test_dace_interrupted_cache_hit_keeps_library(clean_build_folder, monkeypatch):
    """A compile step interrupted while dace reuses the complete library must leave the
    build marked complete. Otherwise the next compile step deletes and rebuilds a library
    that other processes may be loading at that moment."""
    inp = _make_input("interrupted_cache_hit")
    comp = _compiler()
    build_folder = clean_build_folder(comp, inp)

    library_path = comp(inp).library_path
    # A hard link keeps the inode allocated, so a rebuilt library cannot reuse its number.
    pinned_library = build_folder / "pinned_library"
    pinned_library.hardlink_to(library_path)

    def interrupted_compile(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(dace.SDFG, "compile", interrupted_compile)
        with pytest.raises(KeyboardInterrupt):
            comp(inp)

    comp(inp)

    assert library_path.samefile(pinned_library)


def test_dace_build_folder_is_probed_under_lock(clean_build_folder, monkeypatch):
    """dace creates ``FOLDER_MODE`` before it writes the mode into it, and reads an
    empty marker as the unknown mode ``''`` rather than as an absent one. A process
    that probes the build folder while another one holds the lock can therefore see
    the marker half-written and fail, so the compile step must not look into the
    folder before it holds the lock."""
    inp = _make_input("probed_under_lock")
    comp = _compiler()
    build_folder = clean_build_folder(comp, inp)
    build_folder.mkdir(parents=True)
    folder_mode = build_folder / "FOLDER_MODE"
    folder_mode.write_text("")

    real_lock = locking.lock

    @contextlib.contextmanager
    def lock_held_by_writer(directory):
        # The concurrent writer completes the marker before it releases the lock.
        folder_mode.write_text("production")
        with real_lock(directory):
            yield

    monkeypatch.setattr(dace_wf_compilation.locking, "lock", lock_held_by_writer)

    artifact = comp(inp)

    ctypes.CDLL(str(artifact.library_path))
