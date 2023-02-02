add_subdirectory(cartesian)
add_subdirectory(icosahedral)

gridtools_add_unit_test(test_axis SOURCES test_axis.cpp)
gridtools_add_unit_test(test_grid SOURCES test_grid.cpp)
