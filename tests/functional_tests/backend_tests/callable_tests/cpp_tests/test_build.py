import tempfile
import pathlib

import pytest

from functional.backend.callable.cpp import build


@pytest.fixture
def project_input():
    name = "example"
    files = {
        "dllmain.cpp": "void dllmain() {}"
    }
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
