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

# Include any dependencies generated for this target.
include CMakeFiles/copy_stencil_field_view.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/copy_stencil_field_view.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/copy_stencil_field_view.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/copy_stencil_field_view.dir/flags.make

CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o: CMakeFiles/copy_stencil_field_view.dir/flags.make
CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/copy_stencil_field_view_driver.cpp
CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o: CMakeFiles/copy_stencil_field_view.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o -MF CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o.d -o CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o -c /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/copy_stencil_field_view_driver.cpp

CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/copy_stencil_field_view_driver.cpp > CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.i

CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/copy_stencil_field_view_driver.cpp -o CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.s

# Object files for target copy_stencil_field_view
copy_stencil_field_view_OBJECTS = \
"CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o"

# External object files for target copy_stencil_field_view
copy_stencil_field_view_EXTERNAL_OBJECTS =

copy_stencil_field_view: CMakeFiles/copy_stencil_field_view.dir/copy_stencil_field_view_driver.cpp.o
copy_stencil_field_view: CMakeFiles/copy_stencil_field_view.dir/build.make
copy_stencil_field_view: libregression_main.a
copy_stencil_field_view: lib/libgmock.a
copy_stencil_field_view: lib/libgtest.a
copy_stencil_field_view: CMakeFiles/copy_stencil_field_view.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable copy_stencil_field_view"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/copy_stencil_field_view.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/copy_stencil_field_view.dir/build: copy_stencil_field_view
.PHONY : CMakeFiles/copy_stencil_field_view.dir/build

CMakeFiles/copy_stencil_field_view.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/copy_stencil_field_view.dir/cmake_clean.cmake
.PHONY : CMakeFiles/copy_stencil_field_view.dir/clean

CMakeFiles/copy_stencil_field_view.dir/depend:
	cd /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/CMakeFiles/copy_stencil_field_view.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/copy_stencil_field_view.dir/depend

