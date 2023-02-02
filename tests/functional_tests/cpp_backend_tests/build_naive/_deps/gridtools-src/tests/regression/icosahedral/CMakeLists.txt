
function(gridtools_add_icosahedral_test tgt_name)
    gridtools_add_regression_test(${tgt_name} ${ARGN}
            LIB_PREFIX backend_testee
            KEYS ${GT_STENCILS}
            LABELS icosahedral)
endfunction()

gridtools_add_icosahedral_test(stencil_on_edges_multiplefields SOURCES stencil_on_edges_multiplefields.cpp PERFTEST)
gridtools_add_icosahedral_test(stencil_on_cells SOURCES stencil_on_cells.cpp PERFTEST)
gridtools_add_icosahedral_test(stencil_on_neighcell_of_edges SOURCES stencil_on_neighcell_of_edges.cpp PERFTEST)
gridtools_add_icosahedral_test(stencil_manual_fold SOURCES stencil_manual_fold.cpp PERFTEST)
gridtools_add_icosahedral_test(copy_stencil_icosahedral SOURCES copy_stencil_icosahedral.cpp)
gridtools_add_icosahedral_test(expandable_parameters_icosahedral SOURCES expandable_parameters_icosahedral.cpp)
gridtools_add_icosahedral_test(stencil_on_cells_color SOURCES stencil_on_cells_color.cpp)
gridtools_add_icosahedral_test(stencil_on_edges SOURCES stencil_on_edges.cpp)
gridtools_add_icosahedral_test(stencil_fused SOURCES stencil_fused.cpp)
gridtools_add_icosahedral_test(stencil_on_neighedge_of_cells SOURCES stencil_on_neighedge_of_cells.cpp)
gridtools_add_icosahedral_test(stencil_on_vertices SOURCES stencil_on_vertices.cpp)
gridtools_add_icosahedral_test(curl SOURCES curl.cpp)
gridtools_add_icosahedral_test(div SOURCES div.cpp)
gridtools_add_icosahedral_test(lap SOURCES lap.cpp)
