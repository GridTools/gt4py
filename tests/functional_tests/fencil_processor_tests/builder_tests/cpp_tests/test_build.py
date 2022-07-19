# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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


import pathlib
import tempfile

import pytest

from functional.fencil_processors.builders.cpp import build


@pytest.fixture
def project_input():
    name = "example"
    files = {"dllmain.cpp": "void dllmain() {}"}
    deps = []
    return name, deps, files


def test_cmake_no_folder(project_input):
    name, deps, sources = project_input
    project = build.CMakeProject(name=name, dependencies=deps, sources=sources)
    with pytest.raises(RuntimeError):
        project.configure()


def test_cmake_configure_build(project_input):
    name, deps, sources = project_input
    project = build.CMakeProject(name=name, dependencies=deps, sources=sources)
    with tempfile.TemporaryDirectory() as folder:
        project.write(pathlib.Path(folder))
        project.configure()
        project.build()
        output = project.get_current_binary()
        assert output.exists()
