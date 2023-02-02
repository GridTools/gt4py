# CMake 3.14 doesn't respect CMAKE_CUDA_HOST_COMPILER in check_language (but enable_language does),
# see https://gitlab.kitware.com/cmake/cmake/issues/19013

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckLanguage
-------------

Check if a language can be enabled

Usage:

::

  check_language(<lang>)

where <lang> is a language that may be passed to enable_language()
such as "Fortran".  If CMAKE_<lang>_COMPILER is already defined the
check does nothing.  Otherwise it tries enabling the language in a
test project.  The result is cached in CMAKE_<lang>_COMPILER as the
compiler that was found, or NOTFOUND if the language cannot be
enabled.

Example:

::

  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
  else()
    message(STATUS "No Fortran support")
  endif()
#]=======================================================================]

include_guard(GLOBAL)

macro(_workaround_check_language lang)
  if(NOT DEFINED CMAKE_${lang}_COMPILER)
    set(_desc "Looking for a ${lang} compiler")
    message(STATUS ${_desc})
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Check${lang})
    
    set(_testfile "cmake_minimum_required(VERSION ${CMAKE_VERSION})\n")
    if("${lang}" STREQUAL "CUDA" AND DEFINED CMAKE_CUDA_HOST_COMPILER)
        string(APPEND _testfile "set(CMAKE_CUDA_HOST_COMPILER \"${CMAKE_CUDA_HOST_COMPILER}\")\n")
    endif()
    string(APPEND _testfile "project(Check${lang} ${lang})
    file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/result.cmake\"
          \"set(CMAKE_${lang}_COMPILER \\\"\${CMAKE_${lang}_COMPILER}\\\")\\n\"
            )
        ")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Check${lang}/CMakeLists.txt" ${_testfile})

    if(CMAKE_GENERATOR_INSTANCE)
      set(_D_CMAKE_GENERATOR_INSTANCE "-DCMAKE_GENERATOR_INSTANCE:INTERNAL=${CMAKE_GENERATOR_INSTANCE}")
    else()
      set(_D_CMAKE_GENERATOR_INSTANCE "")
    endif()
    execute_process(
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Check${lang}
      COMMAND ${CMAKE_COMMAND} . -G ${CMAKE_GENERATOR}
                                 -A "${CMAKE_GENERATOR_PLATFORM}"
                                 -T "${CMAKE_GENERATOR_TOOLSET}"
                                 ${_D_CMAKE_GENERATOR_INSTANCE}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE result
      )
    include(${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Check${lang}/result.cmake OPTIONAL)
    if(CMAKE_${lang}_COMPILER AND "${result}" STREQUAL "0")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "${_desc} passed with the following output:\n"
        "${output}\n")
    else()
      set(CMAKE_${lang}_COMPILER NOTFOUND)
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "${_desc} failed with the following output:\n"
        "${output}\n")
    endif()
    message(STATUS "${_desc} - ${CMAKE_${lang}_COMPILER}")
    set(CMAKE_${lang}_COMPILER "${CMAKE_${lang}_COMPILER}" CACHE FILEPATH "${lang} compiler")
    mark_as_advanced(CMAKE_${lang}_COMPILER)
  endif()
endmacro()

