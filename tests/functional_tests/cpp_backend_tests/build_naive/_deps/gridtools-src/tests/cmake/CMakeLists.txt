set(_cmake_test_root_dir ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})

# This default test driver assumes that the project
# - is located in a subdirectory called ${name}
# - generates an executable called 'main' which exits with exit code 0 on success
function(add_cmake_test name)
    set(test_build_dir ${_cmake_test_root_dir}/${name})
    set(test_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/${name})
    set(test_driver_filename ${_cmake_test_root_dir}/test_driver.${name}.sh)
    configure_file(test_driver.sh.in ${test_driver_filename} @ONLY)
    add_test(NAME cmaketest_${name}
        COMMAND /usr/bin/env sh ${test_driver_filename}
    )
    set_tests_properties(cmaketest_${name} PROPERTIES LABELS cmake)
endfunction()

if(TARGET _gridtools_cuda)
    add_cmake_test(storage_gpu)
endif()
