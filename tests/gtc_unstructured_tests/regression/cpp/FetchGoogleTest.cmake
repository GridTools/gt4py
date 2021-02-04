function(fetch_googletest)
    # The gtest library needs to be built as static library to avoid RPATH issues
    set(BUILD_SHARED_LIBS OFF)

    include(FetchContent)
    option(INSTALL_GTEST OFF)
    mark_as_advanced(INSTALL_GTEST)
    FetchContent_Declare(
        GoogleTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
    )
    FetchContent_MakeAvailable(GoogleTest)
endfunction()
