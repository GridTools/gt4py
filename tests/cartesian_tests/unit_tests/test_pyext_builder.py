# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests that generated extension modules are not compiled with CPython's own build flags.

CPython records flags such as `-g -O3 -DNDEBUG` in its build configuration and
`distutils` hands them to every extension module it builds. GT4Py has to strip
them, `-g` above all: debug info made the stencil cache of a real model grow
from 647 MB to 21 GB. These tests assert on the command lists of the configured
compiler and on the flags it is handed for a compilation, so they stay
meaningful independently of where setuptools reads its build configuration from.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gt4py.cartesian.backend import pyext_builder


def _optimization_flags(command: list[str]) -> list[str]:
    """Returns the debug-info, optimization and profiling flags of a command line."""
    return [
        arg
        for arg in command
        if arg.startswith(("-g", "-O")) or arg in ("-DNDEBUG", "-pg", "-Wstrict-prototypes")
    ]


class _RecordingBuildExtension(pyext_builder.BuildExtension):
    """Records the state of the configured compiler and the flags of a compilation."""

    #: Command lists of the compiler, keyed as in its 'executables' mapping.
    compiler_commands: dict[str, list[str]]
    #: Flags of the single compilation, in the order the compiler receives them.
    compile_flags: list[str]

    def build_extensions(self) -> None:
        original_compile = self.compiler._compile

        def recording_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # The compiler prepends its own command list and appends 'extra_postargs',
            # which is where the flags of 'get_gt_pyext_build_opts' end up.
            type(self).compile_flags = [
                *self.compiler.compiler_so_cxx,
                *cc_args,
                *extra_postargs,
            ]
            original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = recording_compile
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = original_compile
        type(self).compiler_commands = {
            name: command
            for name in self.compiler.executables
            if isinstance(command := getattr(self.compiler, name, None), list)
        }


def _build_trivial_extension(
    tmp_path: pathlib.Path, **build_opts_kwargs: Any
) -> type[_RecordingBuildExtension]:
    """Builds a minimal extension module and returns the recording 'build_ext' class."""
    source = tmp_path / "trivial.cpp"
    source.write_text('extern "C" int gt4py_trivial() { return 0; }\n', encoding="utf-8")

    recorder = type("_Recorder", (_RecordingBuildExtension,), {})
    pyext_builder.build_pybind_ext(
        "gt4py_trivial_ext",
        [str(source)],
        str(tmp_path / "build"),
        str(tmp_path / "target"),
        build_ext_class=recorder,
        **pyext_builder.get_gt_pyext_build_opts(**build_opts_kwargs),
    )
    return recorder


@pytest.fixture(scope="module")
def default_build(tmp_path_factory) -> type[_RecordingBuildExtension]:
    return _build_trivial_extension(tmp_path_factory.mktemp("default"))


@pytest.fixture(scope="module")
def debug_build(tmp_path_factory) -> type[_RecordingBuildExtension]:
    return _build_trivial_extension(tmp_path_factory.mktemp("debug"), debug_mode=True)


def test_compiler_commands_are_scrubbed(default_build) -> None:
    """None of the compiler's command lists carries CPython's own build flags."""
    assert default_build.compiler_commands, "no compiler command lists were recorded"
    for name, command in default_build.compiler_commands.items():
        assert _optimization_flags(command) == [], (
            f"compiler.{name} still carries {_optimization_flags(command)}"
        )


def test_compile_flags_have_no_debug_info(default_build) -> None:
    """No '-g' is handed to the compiler, so object files carry no DWARF debug info."""
    assert [arg for arg in default_build.compile_flags if arg.startswith("-g")] == []


def test_compile_flags_keep_gt4py_optimization(default_build) -> None:
    """GT4Py's own '-O3 -DNDEBUG' survive the scrubbing and are the only such flags."""
    assert _optimization_flags(default_build.compile_flags) == ["-O3", "-DNDEBUG"]


def test_debug_mode_compile_flags(debug_build) -> None:
    """'debug_mode=True' still yields an unoptimized build with debug info."""
    assert _optimization_flags(debug_build.compile_flags) == ["-O0", "-ggdb"]


def test_cuda_build_extension_scrubs_flags(tmp_path) -> None:
    """
    'CUDABuildExtension' inherits the scrubbing and still dispatches per source type.

    The GPU backends pass 'extra_compile_args' as a dict keyed by source language;
    this builds a C++ source so the test needs no CUDA toolchain, while still
    covering that the subclass routes through 'BuildExtension.build_extensions'.
    """
    recorder = type(
        "_CudaRecorder", (_RecordingBuildExtension, pyext_builder.CUDABuildExtension), {}
    )
    source = tmp_path / "trivial.cpp"
    source.write_text('extern "C" int gt4py_trivial() { return 0; }\n', encoding="utf-8")

    pyext_builder.build_pybind_ext(
        "gt4py_cuda_trivial_ext",
        [str(source)],
        str(tmp_path / "build"),
        str(tmp_path / "target"),
        extra_compile_args={"cxx": ["-std=c++20", "-O3", "-DNDEBUG"], "cuda": ["-O3", "-DNDEBUG"]},
        extra_link_args=["-O3", "-DNDEBUG"],
        build_ext_class=recorder,
    )

    assert _optimization_flags(recorder.compile_flags) == ["-O3", "-DNDEBUG"]
    for name, command in recorder.compiler_commands.items():
        assert _optimization_flags(command) == [], (
            f"compiler.{name} still carries {_optimization_flags(command)}"
        )
