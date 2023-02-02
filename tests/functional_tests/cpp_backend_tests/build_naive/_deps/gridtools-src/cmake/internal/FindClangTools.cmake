# Try to find clang-tidy and clang-format
#
# Usage of this module:
#
#  find_package(ClangTools)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#  ClangTools_PATH      When set, this path is inspected instead of standard library binary 
#                       locations to find clang-tidy and clang-format
#
# This module defines:
#  CLANG_TIDY_BIN       The  path to the clang tidy binary
#  CLANG_TIDY_FOUND     Whether clang tidy was found
#  CLANG_FORMAT_BIN     The path to the clang format binary 
#  CLANG_FORMAT_FOUND   Whether clang format was found
#

find_program(CLANG_TIDY_BIN
    NAMES clang-tidy-6.0.0 clang-tidy-6 clang-tidy
    DOC "Path to clang tidy executable"
    PATHS ${ClangTools_PATH} $ENV{CLANG_TOOLS_PATH}
)
if(NOT CLANG_TIDY_BIN)
    set(CLANG_TIDY_FOUND 0)
    message(STATUS "clang-tidy not found")
else()
    set(CLANG_TIDY_FOUND 1)
    message(STATUS "clang-tidy found at ${CLANG_TIDY_BIN}")
    mark_as_advanced(CLANG_TIDY_BIN)
endif()

find_program(CLANG_FORMAT_BIN
    NAMES clang-format-6.0.0 clang-format-6 clang-format
    DOC "Path to clang format executable"
    PATHS ${ClangTools_PATH} $ENV{CLANG_TOOLS_PATH}
    )
if(NOT CLANG_FORMAT_BIN)
    set(CLANG_FORMAT_FOUND 0)
    message(STATUS "clang-format not found")
else()
    set(CLANG_FORMAT_FOUND 1)
    message(STATUS "clang-format found at ${CLANG_FORMAT_BIN}")
    mark_as_advanced(CLANG_FORMAT_BIN)
endif()
