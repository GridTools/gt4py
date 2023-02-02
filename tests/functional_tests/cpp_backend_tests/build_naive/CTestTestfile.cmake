# CMake generated Testfile for 
# Source directory: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests
# Build directory: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(copy_stencil "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/copy_stencil")
set_tests_properties(copy_stencil PROPERTIES  _BACKTRACE_TRIPLES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;61;add_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;71;add_fn_codegen_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;0;")
add_test(copy_stencil_field_view "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/copy_stencil_field_view")
set_tests_properties(copy_stencil_field_view PROPERTIES  _BACKTRACE_TRIPLES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;61;add_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;72;add_fn_codegen_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;0;")
add_test(anton_lap "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/anton_lap")
set_tests_properties(anton_lap PROPERTIES  _BACKTRACE_TRIPLES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;61;add_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;73;add_fn_codegen_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;0;")
add_test(fvm_nabla "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/fvm_nabla")
set_tests_properties(fvm_nabla PROPERTIES  _BACKTRACE_TRIPLES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;61;add_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;74;add_fn_codegen_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;0;")
add_test(tridiagonal_solve "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/tridiagonal_solve")
set_tests_properties(tridiagonal_solve PROPERTIES  _BACKTRACE_TRIPLES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;61;add_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;75;add_fn_codegen_test;/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/CMakeLists.txt;0;")
subdirs("_deps/gridtools-build")
subdirs("_deps/googletest-build")
