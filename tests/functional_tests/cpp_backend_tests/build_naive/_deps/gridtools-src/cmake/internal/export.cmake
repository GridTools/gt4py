include(CMakePackageConfigHelpers)

# for install tree
set(GRIDTOOLS_MODULE_PATH lib/cmake/GridTools)
configure_package_config_file(cmake/internal/GridToolsConfig.cmake.in
    ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake
    PATH_VARS GRIDTOOLS_MODULE_PATH
    INSTALL_DESTINATION lib/cmake/GridTools)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install(DIRECTORY include/gridtools/ DESTINATION include/gridtools)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/public/ DESTINATION "lib/cmake/${PROJECT_NAME}")

install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake"
              "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake"
        DESTINATION "lib/cmake/${PROJECT_NAME}"
        )

install(EXPORT GridToolsTargets
    FILE GridToolsTargets.cmake
    NAMESPACE GridTools::
    DESTINATION "lib/cmake/${PROJECT_NAME}"
    )

# for build tree
# this registers the build-tree with a global CMake-registry
export(PACKAGE GridTools)
set(GRIDTOOLS_MODULE_PATH ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake)
configure_package_config_file(cmake/internal/GridToolsConfig.cmake.in
    ${PROJECT_BINARY_DIR}/GridToolsConfig.cmake
    PATH_VARS GRIDTOOLS_MODULE_PATH
    INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
    )
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/GridToolsConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
    )

export(EXPORT GridToolsTargets
    FILE ${PROJECT_BINARY_DIR}/GridToolsTargets.cmake
    NAMESPACE GridTools::
    )

file(COPY ${PROJECT_SOURCE_DIR}/cmake/public/ DESTINATION "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake")
