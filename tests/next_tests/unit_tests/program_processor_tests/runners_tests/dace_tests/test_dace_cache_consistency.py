# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Crash-consistency of the dace build-folder cache.

With ``compiler.use_cache=True`` (set by the dace workflow) dace reuses a build
folder whenever the compiled library merely *exists* and never validates it. An
``sdfg.compile()`` interrupted mid-link leaves a truncated, unloadable
``lib<name>.so`` behind; the next run accepts the HIT and every subsequent
``load()`` crashes on ``dlopen`` until the folder is cleaned manually.

The compile step therefore records a completion marker after each successful
compile; a library without the marker is treated as stale and dropped, forcing
a rebuild.
"""

import ctypes
import pathlib
import shutil

import pytest


dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import config, fingerprinting
from gt4py.next.otf import code_specs, stages
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


def _make_input(name: str) -> stages.ExtensionSource:
    sdfg = _make_compilable_sdfg(name)
    program_source = stages.ProgramSource(
        entry_point=interface.Function(name=sdfg.name, parameters=()),
        source_code=sdfg.to_json(),
        library_deps=(),
        code_spec=code_specs.SDFGCodeSpec(),
    )
    return stages.ExtensionSource(
        program_source=program_source,
        binding_source=stages.BindingSource(
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
    comp: dace_wf_compilation.DaCeCompiler, inp: stages.ExtensionSource
) -> pathlib.Path:
    return gtx_cache.get_cache_folder(
        inp,
        config.BuildCacheLifetime.SESSION,
        build_context_id=fingerprinting.strict_fingerprinter(comp.dace_config_nondefaults),
    )


@pytest.fixture
def clean_build_folder(request):
    def factory(
        comp: dace_wf_compilation.DaCeCompiler, inp: stages.ExtensionSource
    ) -> pathlib.Path:
        folder = _build_folder(comp, inp)
        shutil.rmtree(folder, ignore_errors=True)
        request.addfinalizer(lambda: shutil.rmtree(folder, ignore_errors=True))
        return folder

    return factory


def test_dace_recovers_from_truncated_library(clean_build_folder):
    """F4: a truncated (but present) ``lib<name>.so`` is accepted by dace's
    existence-only cache hit. The compile step must detect the incomplete build
    (missing completion marker) and rebuild instead of handing out a library
    that fails to ``dlopen``."""
    inp = _make_input("f4_truncated")
    comp = _compiler()
    build_folder = clean_build_folder(comp, inp)

    artifact = comp(inp)
    assert artifact.library_path.is_file()

    # Simulate a build interrupted mid-link: truncated library, no completion marker.
    artifact.library_path.write_bytes(b"\x00" * 64)
    (build_folder / dace_wf_compilation._COMPILE_COMPLETE_MARKER).unlink()

    recovered = comp(inp)

    ctypes.CDLL(str(recovered.library_path))  # raises OSError if still truncated
