# Install script for directory: /Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gridtools" TYPE DIRECTORY FILES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-src/include/gridtools/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools" TYPE DIRECTORY FILES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-src/cmake/public/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools" TYPE FILE FILES
    "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build/CMakeFiles/install/GridToolsConfig.cmake"
    "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build/CMakeFiles/install/GridToolsConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools/GridToolsTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools/GridToolsTargets.cmake"
         "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build/CMakeFiles/Export/f622928dc033f0d78b26287653fc86b2/GridToolsTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools/GridToolsTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools/GridToolsTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GridTools" TYPE FILE FILES "/Users/nicolettafarabullini/PycharmProjects/gt4py/tests/functional_tests/cpp_backend_tests/build_naive/_deps/gridtools-build/CMakeFiles/Export/f622928dc033f0d78b26287653fc86b2/GridToolsTargets.cmake")
endif()

