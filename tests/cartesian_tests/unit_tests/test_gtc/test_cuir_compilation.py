# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import gridtools_cpp
import pytest
import setuptools

from gt4py.cartesian.backend import pyext_builder
from gt4py.cartesian.gtc.cuir import cuir
from gt4py.cartesian.gtc.cuir.cuir_codegen import CUIRCodegen

from .cuir_utils import KernelFactory, ProgramFactory
from .utils import match


def build_gridtools_test(tmp_path: Path, code: str):
    tmp_src = tmp_path / "test.cu"
    tmp_src.write_text(code)

    opts = pyext_builder.get_gt_pyext_build_opts(uses_cuda=True)
    assert isinstance(opts["include_dirs"], list)
    opts["include_dirs"].append(gridtools_cpp.get_include_dir())
    ext_module = setuptools.Extension("test", [str(tmp_src.absolute())], language="c++", **opts)
    args = [
        "build_ext",
        "--build-temp=" + str(tmp_src.parent),
        "--build-lib=" + str(tmp_src.parent),
        "--force",
    ]
    setuptools.setup(
        name="test",
        ext_modules=[ext_module],
        script_args=args,
        cmdclass={"build_ext": pyext_builder.CUDABuildExtension},
    )


def make_compilation_input_and_expected():
    return [
        (ProgramFactory(name="test", kernels=[]), r"auto test"),
        (ProgramFactory(name="test", kernels__0=KernelFactory()), r"struct kernel"),
    ]


@pytest.mark.requires_gpu
@pytest.mark.parametrize("cuir_program,expected_regex", make_compilation_input_and_expected())
def test_program_compilation_succeeds(tmp_path, cuir_program, expected_regex):
    assert isinstance(cuir_program, cuir.Program)
    code = CUIRCodegen.apply(cuir_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, expected_regex)
    build_gridtools_test(tmp_path, code)
