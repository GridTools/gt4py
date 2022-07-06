import ctypes

import pytest

from eve.codegen import format_source
from functional.fencil_processors import defs
from functional.fencil_processors.callables.cpp import bindings


@pytest.fixture
def example_function():
    return defs.Function(
        name="example",
        parameters=[
            defs.BufferParameter("buf", 2, ctypes.c_float),
            defs.ScalarParameter("sc", ctypes.c_float),
        ],
    )


def test_bindings(example_function):
    module = bindings.create_bindings(example_function, "example.hpp")
    expected_src = format_source(
        "cpp",
        """\
        #include "example.hpp"
        
        #include <gridtools/common/defs.hpp>
        #include <gridtools/fn/backend/naive.hpp>
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/fn/unstructured.hpp>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        
        decltype(auto) example_wrapper(pybind11::buffer buf, float sc) {
          return example(
              gridtools::as_sid<float, 2, gridtools::integral_constant<int, 0>,
                                999'999'999>(buf),
              sc);
        }
        
        PYBIND11_MODULE(example, module) {
          module.doc() = "";
          module.def("example", &example_wrapper, "");
        }\
        """,
        style="LLVM",
    )
    assert module.library_deps[0].name == "pybind11"
    assert module.source_code == expected_src
