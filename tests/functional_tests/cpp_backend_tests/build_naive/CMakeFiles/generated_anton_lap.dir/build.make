# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_COMMAND = /Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive

# Utility rule file for generated_anton_lap.

# Include any custom commands dependencies for this target.
include CMakeFiles/generated_anton_lap.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/generated_anton_lap.dir/progress.make

CMakeFiles/generated_anton_lap: generated_anton_lap.hpp

generated_anton_lap.hpp: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating generated_anton_lap.hpp"
	/Users/nicolettafarabullini/PycharmProjects/gt4py/.venv/bin/python3.10 /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/anton_lap.py /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/generated_anton_lap.hpp

generated_anton_lap: CMakeFiles/generated_anton_lap
generated_anton_lap: generated_anton_lap.hpp
generated_anton_lap: CMakeFiles/generated_anton_lap.dir/build.make
.PHONY : generated_anton_lap

# Rule to build all files generated by this target.
CMakeFiles/generated_anton_lap.dir/build: generated_anton_lap
.PHONY : CMakeFiles/generated_anton_lap.dir/build

CMakeFiles/generated_anton_lap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/generated_anton_lap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/generated_anton_lap.dir/clean

CMakeFiles/generated_anton_lap.dir/depend:
	cd /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles/generated_anton_lap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/generated_anton_lap.dir/depend

