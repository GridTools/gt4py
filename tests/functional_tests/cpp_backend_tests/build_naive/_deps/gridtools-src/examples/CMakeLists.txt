# This file is part of the GridTools source tree to drive installation of the examples.

function(install_example)
    set(options)
    # - if needed later split into cpu and gpu sources (then add extension)
    set(one_value_args DIRECTORY)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARG "${options}" "${one_value_args};" "${multi_value_args}" ${ARGN})

    string(REPLACE ";" " " EXAMPLES_SRCFILES "${ARG_SOURCES}")
    configure_file(cmake_skeletons/CMakeLists.txt.example.in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ARG_DIRECTORY}/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ARG_DIRECTORY}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/${ARG_DIRECTORY})
    install(
        DIRECTORY ${ARG_DIRECTORY}
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
        )

    list(APPEND enabled_examples ${ARG_DIRECTORY})
    set(enabled_examples ${enabled_examples} PARENT_SCOPE)
endfunction()

if(GT_INSTALL_EXAMPLES)
    set(enabled_examples) # used in CMakeLists.txt.in, filled in install_example()
    install_example(DIRECTORY stencil SOURCES copy_stencil horizontal_diffusion_limited tridiagonal_solver)

    configure_file(stencil_type_erasure/CMakeLists.txt.in
        ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/stencil_type_erasure/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/stencil_type_erasure/CMakeLists.txt
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/stencil_type_erasure)
    install(
        DIRECTORY stencil_type_erasure
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
        PATTERN "CMakeLists.txt.in" EXCLUDE
        )
    list(APPEND enabled_examples stencil_type_erasure)

    install_example(DIRECTORY boundaries SOURCES boundaries boundaries_provided)

    configure_file(c_bindings/CMakeLists.txt.in
        ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/c_bindings/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/c_bindings/CMakeLists.txt
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}/c_bindings)
    install(
        DIRECTORY c_bindings
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
        PATTERN "CMakeLists.txt.in" EXCLUDE
        )
    list(APPEND enabled_examples c_bindings)

    configure_file(cmake_skeletons/CMakeLists.txt.driver.in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeLists.txt @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeLists.txt DESTINATION ${GT_INSTALL_EXAMPLES_PATH})

    install(
        FILES
            ${PROJECT_SOURCE_DIR}/cmake/public/workaround_check_language.cmake
            ${PROJECT_SOURCE_DIR}/cmake/public/detect_features.cmake
            ${PROJECT_SOURCE_DIR}/cmake/public/try_compile_clang_cuda.cmake
            ${PROJECT_SOURCE_DIR}/cmake/public/try_compile_hip.cmake
        DESTINATION ${GT_INSTALL_EXAMPLES_PATH}
    )
endif()
