# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-src"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/tmp"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/src/gridtools-populate-stamp"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/src"
  "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/src/gridtools-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/src/gridtools-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-subbuild/gridtools-populate-prefix/src/gridtools-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
