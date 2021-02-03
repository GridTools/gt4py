#include <array>
#include <gridtools/meta/at.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>
#include <gridtools/sid/concept.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

template <typename NeighborChain, typename Mesh>
auto get_neighbors(Mesh const &mesh, int i) {
    namespace gs = gridtools::sid;
    using namespace gridtools::next;

    auto conn = mesh::connectivity<NeighborChain>(mesh);

    auto tbl = gridtools::next::connectivity::neighbor_table(conn);
    auto ptr = gs::get_origin(tbl)();
    auto strides = gs::get_strides(tbl);
    gs::shift(ptr, gridtools::at_key<gridtools::meta::at_c<NeighborChain, 0>>(strides), i);
    std::array<int, decltype(gridtools::next::connectivity::max_neighbors(conn))::value> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = *ptr;
        gs::shift(ptr, gridtools::at_key<neighbor>(strides), 1);
    }
    return result;
}

TEST(simple_mesh, cell2cell) {
    gridtools::next::test_helper::simple_mesh mesh;

    auto c2c = gridtools::next::mesh::connectivity<std::tuple<cell, cell>>(mesh);

    ASSERT_EQ(9, gridtools::next::connectivity::size(c2c));
    ASSERT_EQ(4, gridtools::next::connectivity::max_neighbors(c2c));
    ASSERT_EQ(-1, gridtools::next::connectivity::skip_value(c2c));

    ASSERT_THAT((get_neighbors<std::tuple<cell, cell>>(mesh, 0)), testing::UnorderedElementsAre(1, 3, 2, 6));
    ASSERT_THAT((get_neighbors<std::tuple<cell, cell>>(mesh, 1)), testing::UnorderedElementsAre(0, 7, 2, 4));
    // etc
}

TEST(simple_mesh, edge2vertex) {
    gridtools::next::test_helper::simple_mesh mesh;

    auto e2v = gridtools::next::mesh::connectivity<std::tuple<edge, vertex>>(mesh);

    ASSERT_EQ(18, gridtools::next::connectivity::size(e2v));
    ASSERT_EQ(2, gridtools::next::connectivity::max_neighbors(e2v));
    ASSERT_EQ(-1, gridtools::next::connectivity::skip_value(e2v));

    ASSERT_THAT((get_neighbors<std::tuple<edge, vertex>>(mesh, 0)), testing::UnorderedElementsAre(0, 1));
    // etc
}
