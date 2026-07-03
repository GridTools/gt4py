# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Crash-consistency of the dace build-folder cache.

With ``compiler.use_cache=True`` (set by the dace workflow) DaCe reuses a build
folder whenever the compiled library merely *exists* (``lib_path.is_file()``) and
then ``dlopen``s it. An ``sdfg.compile()`` interrupted mid-link leaves a
truncated, unloadable ``lib<name>.so`` behind; the next run accepts the HIT and
crashes on load instead of rebuilding.

A *missing* library already self-heals (``is_file()`` is False, and DaCe
regenerates because ``SDFG.regenerate_code`` defaults to True), so only the
truncated-but-present case is a bug. Reproducing it in-process requires releasing
the ``dlopen`` handle first (via ``ReloadableDLL.unload()``): otherwise glibc
returns the cached handle and never re-reads the corrupted file from disk -- which
is exactly why the failure only bites on a fresh "re-run" process in practice.
"""

import shutil

import pytest


dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import config
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


def _make_input(name: str) -> stages.CompilableProject:
    sdfg = _make_compilable_sdfg(name)
    program_source = stages.ProgramSource(
        entry_point=interface.Function(name=sdfg.name, parameters=()),
        source_code=sdfg.to_json(),
        library_deps=(),
        code_spec=code_specs.SDFGCodeSpec(),
    )
    return stages.CompilableProject(
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


@pytest.fixture
def fresh_dace_input():
    """Yield a factory returning a CompilableProject with a wiped build folder.

    Each program gets a unique name (hence a unique build folder), wiped before
    and after, so a partial artifact from an earlier run never leaks in.
    """
    folders = []

    def factory(name: str) -> stages.CompilableProject:
        inp = _make_input(name)
        folder = gtx_cache.get_cache_folder(inp, config.BuildCacheLifetime.SESSION)
        if folder.exists():
            shutil.rmtree(folder)
        folders.append(folder)
        return inp

    yield factory

    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)


def test_dace_recovers_from_truncated_library(fresh_dace_input):
    """F4: a truncated (but present) ``lib<name>.so`` is accepted by DaCe's
    ``is_file()`` cache hit and ``dlopen``ed. The compile step must recover by
    rebuilding instead of crashing on load."""
    name = "f4_truncated"
    inp = fresh_dace_input(name)
    comp = _compiler()

    program = comp(inp)  # real compile -> the .so is loaded into this process
    program.sdfg_program._lib.unload()  # release the dlopen handle so the file is re-read
    del program

    build_folder = gtx_cache.get_cache_folder(inp, config.BuildCacheLifetime.SESSION)
    library = build_folder / f"lib{name}.so"
    assert library.is_file()
    library.write_bytes(b"\x00" * 64)  # truncate -> present but unloadable

    recovered = comp(inp)  # must rebuild, not raise OSError on dlopen
    assert recovered.sdfg_program is not None
