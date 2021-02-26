# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

import pytest
import setuptools

from gt4py import (  # TODO(havogt) this is a dependency from gtc tests to gt4py, ok?
    config,
    gt_src_manager,
)
from gt4py.backend import pyext_builder
from gtc.cuir import cuir
from gtc.cuir.cuir_codegen import CUIRCodegen

from .cuir_utils import ProgramFactory
from .utils import match


if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
    raise RuntimeError("Missing GridTools sources.")


def build_gridtools_test(tmp_path: Path, code: str):
    tmp_src = tmp_path / "test.cu"
    tmp_src.write_text(code)

    opts = pyext_builder.get_gt_pyext_build_opts(uses_cuda=True)
    assert isinstance(opts["include_dirs"], list)
    opts["include_dirs"].append(config.GT2_INCLUDE_PATH)
    ext_module = setuptools.Extension(
        "test",
        [str(tmp_src.absolute())],
        language="c++",
        **opts,
    )
    args = [
        "build_ext",
        "--build-temp=" + str(tmp_src.parent),
        "--build-lib=" + str(tmp_src.parent),
        "--force",
    ]
    setuptools.setup(
        name="test",
        ext_modules=[
            ext_module,
        ],
        script_args=args,
        cmdclass={"build_ext": pyext_builder.CUDABuildExtension},
    )


def make_compilation_input_and_expected():
    return [
        (ProgramFactory(name="test", kernels=[]), r"auto test"),
    ]


@pytest.mark.parametrize("cuir_program,expected_regex", make_compilation_input_and_expected())
def test_program_compilation_succeeds(tmp_path, cuir_program, expected_regex):
    assert isinstance(cuir_program, cuir.Program)
    code = CUIRCodegen.apply(cuir_program, gt_backend_t="cpu_ifirst")
    print(code)
    match(code, expected_regex)
    build_gridtools_test(tmp_path, code)
