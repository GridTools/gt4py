# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/lib/python3.10/site-packages/cmake/data/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/lib/python3.10/site-packages/cmake/data/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive

# Utility rule file for generated_copy_stencil.

# Include any custom commands dependencies for this target.
include CMakeFiles/generated_copy_stencil.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/generated_copy_stencil.dir/progress.make

CMakeFiles/generated_copy_stencil: generated_copy_stencil.hpp

generated_copy_stencil.hpp: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating generated_copy_stencil.hpp"
	/Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/bin/python3.10 /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/copy_stencil.py /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/generated_copy_stencil.hpp

generated_copy_stencil: CMakeFiles/generated_copy_stencil
generated_copy_stencil: generated_copy_stencil.hpp
generated_copy_stencil: CMakeFiles/generated_copy_stencil.dir/build.make
.PHONY : generated_copy_stencil

# Rule to build all files generated by this target.
CMakeFiles/generated_copy_stencil.dir/build: generated_copy_stencil
.PHONY : CMakeFiles/generated_copy_stencil.dir/build

CMakeFiles/generated_copy_stencil.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/generated_copy_stencil.dir/cmake_clean.cmake
.PHONY : CMakeFiles/generated_copy_stencil.dir/clean

CMakeFiles/generated_copy_stencil.dir/depend:
	cd /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles/generated_copy_stencil.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/generated_copy_stencil.dir/depend

